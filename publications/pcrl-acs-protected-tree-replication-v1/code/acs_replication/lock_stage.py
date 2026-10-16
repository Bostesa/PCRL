"""Stage locks: TRAINING_COMPLETE_LOCK (all banks frozen before MODEL_SELECT selection) and REPORTING_LOCK
(all reader banks and DEV_REPORT scores frozen before DEV_REPORT truth opens). Hashes only; no label is read here.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
RUNS = ROOT / '.private' / 'acs_replication' / 'runs'


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_hashes(prefixes):
    out = {}
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and d.name.startswith(prefixes):
            for p in sorted(d.rglob('*')):
                if p.is_file():
                    out[str(p.relative_to(RUNS))] = sha(p)
    return out


def training_complete(notes):
    rec = {'schema': 'ACSR1-training-complete-lock-v1', 'status': 'TRAINING_COMPLETE_LOCKED', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'frozen_files_sha256': tree_hashes(('STAGE_A_', 'STAGE_B_', 'NEURAL_RAW_', 'FARE_', 'FAGTB_')),
           'records': {}, 'notes': notes, 'MODEL_SELECT_labels_opened_for_selection': False, 'DEV_REPORT_labels_opened': False}
    for d in sorted(RUNS.iterdir()):
        for name in ('STAGE_A_SEED.json', 'STAGE_B_SEED.json', 'NEURAL_RAW_SEED.json', 'FARE_SEED.json', 'FAGTB_SEED.json'):
            p = d / name
            if p.is_file():
                rec['records'][f'{d.name}/{name}'] = sha(p)
    p = RESULTS / 'TRAINING_COMPLETE_LOCK.json'
    with open(p, 'x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return p


def reporting(notes):
    sel = RESULTS / 'SELECTION_LOCK.json'
    rec = {'schema': 'ACSR1-reporting-lock-v1', 'status': 'REPORTING_LOCKED', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'selection_lock_sha256': sha(sel), 'reader_files_sha256': tree_hashes(('READERS_',)), 'notes': notes,
           'readers_frozen_before_DEV_REPORT_truth': True, 'DEV_REPORT_labels_opened': False, 'sealed_assessment_touched': False}
    for s in (0, 1):
        r = json.loads((RUNS / f'READERS_{s}' / 'READERS_SEED.json').read_text())
        rec[f'winners_s{s}'] = {k: {v: w['reader'] for v, w in sv['winner'].items()} for k, sv in r['services'].items()}
        rec[f'dev_report_scores_sha256_s{s}'] = r['dev_report_scores_sha256']
    p = RESULTS / 'REPORTING_LOCK.json'
    with open(p, 'x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return p


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['training', 'reporting']); ap.add_argument('--notes', default='')
    a = ap.parse_args()
    p = training_complete(a.notes) if a.which == 'training' else reporting(a.notes)
    print(json.dumps({'path': str(p), 'sha256': sha(p)}))
