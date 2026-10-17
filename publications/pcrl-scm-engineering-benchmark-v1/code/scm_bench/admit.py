"""Admission record: the stopped structural campaign's artifacts used by this follow-up, identified by hash.

Writes results/pcrl_scm_engineering_benchmark_v1/ADMISSION.json. Reads text records only (no array is loaded).
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

from . import common as C

SRC_RES = 'results/pcrl_structural_mechanism_pilot_v1'
FILES = ['SERVICE_AND_THREAT_MODEL.md', 'SELECTION_DECISION.md', 'START_HERE.md', 'roles/designer_1/PROPOSAL.md',
         'roles/prior_work_reviewer/FIELD_MATRIX.md', 'roles/prior_work_reviewer/PROPOSAL_COMPARISON.md', 'roles/verifier/VERIFICATION.md',
         'roles/adversary_reviewer/PROPOSAL_THREAT_CHECK.md', 'HANDOFF.json']


def git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def main():
    tip = git('rev-parse', 'research/pcrl-structural-mechanism-pilot-v1')
    rec = {'schema': 'SCM1-admission-v1', 'utc': C.now(), 'source_branch': 'research/pcrl-structural-mechanism-pilot-v1', 'source_tip': tip,
           'source_tip_expected': 'ec20c8bdf', 'source_tip_matches': tip.startswith('ec20c8bdf'),
           'public_package': {'branch': 'publication/pcrl-structural-mechanism-pilot-v1',
                              'remote_sha': git('ls-remote', 'origin', 'refs/heads/publication/pcrl-structural-mechanism-pilot-v1').split('\t')[0]},
           'files_at_source_tip_sha256': {}, 'private_records_sha256': {}}
    for f in FILES:
        blob = subprocess.run(['git', '-C', str(C.ROOT), 'show', f'{tip}:{SRC_RES}/{f}'], capture_output=True).stdout
        rec['files_at_source_tip_sha256'][f] = hashlib.sha256(blob).hexdigest()
    sp = C.ROOT.parents[0] / 'pcrl-structural-mechanism-pilot-v1' / '.private' / 'pcrl_structural_mechanism_pilot_v1'
    for name in ('RESTORE_RECEIPT.json', 'compute/ledger.jsonl'):
        p = sp / name
        rec['private_records_sha256'][name] = C.sha(p) if p.is_file() else 'MISSING'
    rec['this_followup_restore_receipt_sha256'] = C.sha(C.PRIV / 'RESTORE_RECEIPT.json')
    rec['note'] = ('the stopped campaign restored the same 38 source files with the same published hashes; this follow-up restored them again '
                   'from the source worktrees (unit SCM_RESTORE_V1) rather than copying the stopped campaign\'s private copies')
    rec['proposal_check'] = rec['files_at_source_tip_sha256']['roles/designer_1/PROPOSAL.md'] == '6bc0617b286dccd2e7b428c51b15c6399696cacf5737415188d6db46f0aa4002'
    (C.RESULTS / 'ADMISSION.json').write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({k: rec[k] for k in ('source_tip', 'source_tip_matches', 'public_package', 'proposal_check')}, indent=1))


if __name__ == '__main__':
    main()
