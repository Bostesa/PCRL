"""Verifier wave 3 (label-free, standard library only): leak scan of a sanitized publication tree (read as data; nothing in it is run).

Per file: binary / array files (extension or NUL / non-UTF-8 content); every line matching personal paths (/Users/, the user name,
/Volumes/, worktree paths), e-mail addresses, tokens or keys; and record-level data (long numeric arrays in JSON, JSON keys that name
per-person fields with long list values, CSV / TSV tables with person-level columns, long runs of numbers on a line).

    python3 -I .../roles/verifier/code/v3_pubscan.py --root <publication dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BINARY_EXT = {'.npz', '.npy', '.pkl', '.pickle', '.pt', '.pth', '.joblib', '.h5', '.hdf5', '.parquet', '.feather', '.zip', '.gz', '.tar',
              '.bz2', '.xz', '.7z', '.db', '.sqlite', '.bin', '.so', '.dylib', '.pyc'}
PATTERNS = {
    'personal_path_users': re.compile(r'/Users/'),
    'user_name': re.compile(r'<owner>|<owner>\s*<owner>|<owner>', re.I),
    'volumes_path': re.compile(r'/Volumes/'),
    'worktree_path': re.compile(r'\.claude/worktrees|/\.worktrees/|worktrees/pcrl-'),
    'private_tmp_path': re.compile(r'/private/tmp/|/tmp/claude'),
    'home_tilde_path': re.compile(r'~/'),
    'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'github_token': re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'),
    'aws_key': re.compile(r'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}'),
    'openai_anthropic_key': re.compile(r'sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{10,}'),
    'private_key_block': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'secret_word': re.compile(r'(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|bearer|authorization)\b'),
    'token_word': re.compile(r'(?i)\btoken\b'),
    'repo_owner_account': re.compile(r'Bostesa'),
    'scratchpad_session_path': re.compile(r'claude-501|scratchpad'),
}
PERSON_KEYS = {'row_id', 'row_ids', 'unit', 'units', 'household', 'households_ids', 'sex', 'income', 'occupation', 'label', 'labels', 'y',
               'X', 'agep', 'wkhp', 'cow', 'schl', 'mar', 'pobp', 'raw_inputs', 'serialno', 'sporder', 'pwgtp'}
LONG_NUMS = re.compile(r'(?:-?\d+(?:\.\d+)?(?:e-?\d+)?\s*,\s*){200,}')


def scan_json(obj, path, hits, file):
    if isinstance(obj, dict):
        for k, x in obj.items():
            if isinstance(x, list) and len(x) >= 100 and str(k).lower() in PERSON_KEYS:
                hits.append({'file': file, 'kind': 'json_person_key_long_list', 'where': f'{path}/{k}', 'length': len(x)})
            scan_json(x, f'{path}/{k}', hits, file)
    elif isinstance(obj, list):
        if len(obj) >= 500 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj[:500]):
            hits.append({'file': file, 'kind': 'json_long_numeric_array', 'where': path, 'length': len(obj)})
        for i, x in enumerate(obj[:2000]):
            scan_json(x, f'{path}[{i}]', hits, file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    out = {'root': str(root), 'n_files': len(files), 'files_sha256': {}, 'binary': [], 'line_hits': [], 'record_level': [],
           'csv_tables': [], 'unreadable': []}
    for p in files:
        rel = str(p.relative_to(root))
        b = p.read_bytes()
        out['files_sha256'][rel] = hashlib.sha256(b).hexdigest()
        if p.suffix.lower() in BINARY_EXT or b'\0' in b:
            out['binary'].append({'file': rel, 'bytes': len(b), 'reason': 'extension' if p.suffix.lower() in BINARY_EXT else 'NUL bytes'})
            continue
        try:
            text = b.decode('utf-8')
        except UnicodeDecodeError:
            out['binary'].append({'file': rel, 'bytes': len(b), 'reason': 'not UTF-8'})
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, rx in PATTERNS.items():
                for m in rx.finditer(line):
                    out['line_hits'].append({'file': rel, 'line': i, 'pattern': name, 'match': m.group(0)[:80],
                                             'context': line.strip()[:200]})
            if LONG_NUMS.search(line):
                out['record_level'].append({'file': rel, 'line': i, 'kind': 'long_number_run_on_line', 'context': line.strip()[:120]})
        if p.suffix.lower() == '.json':
            try:
                scan_json(json.loads(text), '', out['record_level'], rel)
            except json.JSONDecodeError as ex:
                out['unreadable'].append({'file': rel, 'error': str(ex)[:120]})
        if p.suffix.lower() in ('.csv', '.tsv'):
            lines = text.splitlines()
            head = [h.strip().strip('"').lower() for h in re.split(r'[,\t]', lines[0])] if lines else []
            out['csv_tables'].append({'file': rel, 'rows': max(len(lines) - 1, 0), 'columns': head[:40],
                                      'person_level_columns': sorted(set(head) & PERSON_KEYS)})
    out['counts'] = {'binary': len(out['binary']), 'line_hits': len(out['line_hits']), 'record_level': len(out['record_level']),
                     'csv_tables': len(out['csv_tables'])}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out['counts']))


if __name__ == '__main__':
    main()
