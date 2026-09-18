"""Write Terminal 2's atomic handoff and the committed aggregate HANDOFF.json.

Atomic: each file is written to a .tmp sibling and renamed, so a reader never sees a
partial record. Writes to the absolute git common directory's
pcrl_parallel_handoff_v3/terminal_2/ and never touches the terminal_1 sibling.

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.write_handoff --repo .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = [
    'papers/pcrl_manuscript_v3/main.tex',
    'papers/pcrl_manuscript_v3/main.pdf',
    'papers/pcrl_manuscript_v3/references.bib',
    'papers/pcrl_manuscript_v3/MANIFEST.json',
    'papers/pcrl_manuscript_v3/DERIVED_FACTS.json',
    'papers/pcrl_manuscript_v3/DERIVED_FACTS_V3.json',
    'results/pcrl_manuscript_review_v3/CLAIM_LEDGER.csv',
    'results/pcrl_manuscript_review_v3/SOURCE_MANIFEST.json',
    'results/pcrl_manuscript_review_v3/CORRECTIONS.md',
    'results/pcrl_manuscript_review_v3/MATHEMATICAL_REVIEW.md',
    'results/pcrl_manuscript_review_v3/RELATED_WORK_COMPARISON.md',
    'results/pcrl_manuscript_review_v3/CONTRIBUTION_ASSESSMENT.md',
    'results/pcrl_manuscript_review_v3/REVIEW_INDEX.md',
    'results/pcrl_manuscript_review_v3/VALIDATION.md',
    'results/pcrl_manuscript_review_v3/VERIFICATION_V3.json',
    'results/pcrl_manuscript_review_v3/PENDING_EXPERIMENT_INTEGRATION.md',
    'results/pcrl_manuscript_review_v3/REPRODUCE_PAPER.md',
    'experiments/pcrl_manuscript_v3/make_assets_v3.py',
    'experiments/pcrl_manuscript_v3/build_claim_ledger_v3.py',
    'experiments/pcrl_manuscript_v3/recheck_study3.py',
    'experiments/pcrl_manuscript_v3/write_handoff.py',
]

BODY = Path(__file__).with_name('handoff_body.json')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    a = ap.parse_args()
    root = Path(a.repo).resolve()

    h = json.loads(BODY.read_text())
    ver = json.loads((root / 'results/pcrl_manuscript_review_v3/VERIFICATION_V3.json').read_text())
    h['timestamp_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    h['head_sha_at_write'] = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    h['artifact_sha256'] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ARTIFACTS}
    h['verification'] = {
        'study3_artifact_hashes_checked': ver['V1_artifact_hashes']['checked'],
        'study3_artifact_hash_mismatches': ver['V1_artifact_hashes']['mismatches'],
        'all_checks_passed': ver['all_passed'],
        'reporting_scope_located_by_reproduction': ver['V3_reporting_scope']['matches'],
        'paired_contrast_scale_invariant': ver['V4_scope_invariance']['contrast_scale_invariant'],
        'new_scientific_fits_run_by_terminal_2': 0,
    }

    common = Path(subprocess.run(['git', 'rev-parse', '--path-format=absolute',
                                  '--git-common-dir'], cwd=root,
                                 capture_output=True, text=True).stdout.strip())
    out = common / 'pcrl_parallel_handoff_v3' / 'terminal_2'
    out.mkdir(parents=True, exist_ok=True)

    body = json.dumps(h, indent=1) + '\n'
    (out / 'HANDOFF.json.tmp').write_text(body)
    (out / 'HANDOFF.json.tmp').replace(out / 'HANDOFF.json')
    status = json.dumps({
        'terminal': 'terminal_2', 'status': h['status'], 'branch': h['branch'],
        'timestamp_utc': h['timestamp_utc'], 'headline': h['headline'],
        'pdf': 'papers/pcrl_manuscript_v3/main.pdf (18 pages)',
        'review': 'results/pcrl_manuscript_review_v3/REVIEW_INDEX.md',
        'new_scientific_fits': 0}, indent=1) + '\n'
    (out / 'STATUS.json.tmp').write_text(status)
    (out / 'STATUS.json.tmp').replace(out / 'STATUS.json')
    (root / 'results/pcrl_manuscript_review_v3/HANDOFF.json').write_text(body)
    print('handoff written to', out)
    print('artifacts hashed:', len(ARTIFACTS))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
