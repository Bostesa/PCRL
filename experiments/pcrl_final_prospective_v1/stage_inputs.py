"""Copy the frozen historical inputs into this study's private root, verifying hashes.

Nothing is fitted or transformed here. Expected hashes come from committed
historical manifests, never from the copied file itself.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path

from .common import INPUTS, ROOT, atomic_json, now, sha

PCRL = Path('/Users/nathansamson/PCRL')
PRED_INPUTS = PCRL/'.worktrees/pcrl-task-directed-release-v1/results/pcrl_task_directed_release_v1/private/inputs'
ADMISSION_DATA = Path('/Users/nathansamson/PCRL-terminal-b/data/acs_2016_admission')
ADMISSION_MANIFEST_COMMIT = '0d8f4b67b6d4961dfa133289d0167c874d2f4794'
ADMISSION_MANIFEST_PATH = 'results/pcrl_evidence_review_v1/ACS_2016_ADMISSION_MANIFEST.json'
SERVICE_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')


def git_show(commit, path):
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'], check=True,
                          capture_output=True).stdout


def expected_hashes():
    out = {}
    restore = json.loads((ROOT/'results/pcrl_task_directed_release_v1/INPUT_RESTORE.json').read_text())
    for record in restore['files']:
        out[record['relative_path']] = (PRED_INPUTS/record['relative_path'], record['sha256'])
    reuse = json.loads((ROOT/'results/redesign_20260909_acs_fixed_predictions_v1/REUSE_MANIFEST.json').read_text())
    identity = json.loads((ROOT/'results/redesign_20260909_acs_fixed_predictions_v1/PREFIT_IDENTITY.json').read_text())
    for seed, record in identity.items():
        for task in SERVICE_TASKS:
            base = record['anchors'][task]['path']
            for name in ('metadata.json', 'preprocessing.npz', 'model.pt', 'model.joblib'):
                rel = f'{base}/{name}'
                if rel in reuse['historical_hashes']:
                    out[rel] = (PCRL/rel, reuse['historical_hashes'][rel])
    import hashlib
    for seed in (0, 1, 2):
        rel = f'results/redesign_20260907_acs_transfer_v1/seed_{seed}/release_maps.joblib'
        out[rel] = (PCRL/rel, reuse['historical_hashes'][rel])
        rel = f'results/redesign_20260907_acs_transfer_v1/seed_{seed}/preprocessing.json'
        tracked = git_show('f4bdf4cd5bf74c634feeec50aef78bff249667e4', rel)
        out[rel] = (PCRL/rel, hashlib.sha256(tracked).hexdigest())
    manifest = json.loads(git_show(ADMISSION_MANIFEST_COMMIT, ADMISSION_MANIFEST_PATH))
    prov = manifest['provenance']['sha256']
    out['data/acs_2016/ss16pca.csv'] = (ADMISSION_DATA/'extracted/ss16pca.csv', prov['ss16pca.csv'])
    out['data/acs_2016/csv_pca_2016.zip'] = (ADMISSION_DATA/'official/csv_pca_2016.zip', prov['csv_pca_2016.zip'])
    out['data/acs_2016/PUMSDataDict16.txt'] = (ADMISSION_DATA/'official/PUMSDataDict16.txt', prov['PUMSDataDict16.txt'])
    out['data/acs_2016/PUMS_Data_Dictionary_2017.txt'] = (ADMISSION_DATA/'official/PUMS_Data_Dictionary_2017.txt',
                                                         prov['PUMS_Data_Dictionary_2017.txt'])
    return out, manifest


def main():
    files, manifest = expected_hashes()
    records = []
    for rel, (source, expected) in sorted(files.items()):
        target = INPUTS/rel
        if not target.exists():
            if sha(source) != expected:
                raise SystemExit(f'Source hash mismatch before staging: {rel}')
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name+'.tmp')
            shutil.copyfile(source, tmp)
            tmp.replace(target)
        actual = sha(target)
        if actual != expected:
            raise SystemExit(f'Staged hash mismatch: {rel}')
        records.append({'path': rel, 'sha256': actual, 'bytes': target.stat().st_size,
                        'source': str(source)})
    (INPUTS/'admission').mkdir(parents=True, exist_ok=True)
    atomic_json(INPUTS/'admission/ACS_2016_ADMISSION_MANIFEST.json', manifest)
    atomic_json(INPUTS/'STAGED_INPUTS.json', {'created_utc': now(), 'files': records,
        'admission_manifest': {'commit': ADMISSION_MANIFEST_COMMIT, 'path': ADMISSION_MANIFEST_PATH},
        'rule': 'expected hashes from committed historical manifests; copies verified after staging'})
    print(len(records), 'staged and verified')


if __name__ == '__main__':
    main()
