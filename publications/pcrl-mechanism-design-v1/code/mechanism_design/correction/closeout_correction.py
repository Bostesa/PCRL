"""Closeout for the correction pass: resource inventory section and the updated handoff.

resources
    Reads the correction pass's own queue ledger (through cqueue) and adds a `correction_pass_2026_10_16` section to
    results/pcrl_mechanism_design_v1/RESOURCE_INVENTORY.json. The campaign's original keys are left unchanged.
handoff EXTRA_JSON
    Writes results/pcrl_mechanism_design_v1/HANDOFF.json (schema MD1-handoff-v2). The superseded v1 handoff is
    identified by its commit and hash, and its full text stays in git history and in public commit e141330.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mechanism_design.correction import cqueue  # noqa: F401  (swaps the queue constants to the correction ledger)
from mechanism_design import closeout as CO
from mechanism_design import jobqueue as Q

ROOT = CO.ROOT
RESULTS = CO.RESULTS
CORRECTION_BASE = '12ffa8558'
NEW_ARTIFACTS = ['CORRECTIONS.md', 'corrections/CHECKS_PREREGISTRATION.md', 'corrections/EXACT_CHECKS.json',
                 'roles/correction_v1/math_reviewer/MATH_REVIEW.md', 'roles/correction_v1/adversary_reviewer/ADVERSARY_REVIEW.md',
                 'roles/correction_v1/adversary_reviewer/MANUSCRIPT_ADVERSARY_PATCH_v2.diff',
                 'roles/correction_v1/verifier/VERIFICATION.md', 'roles/correction_v1/verifier/V_EXACT_CHECKS.json']


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def resources():
    inv_path = RESULTS / 'RESOURCE_INVENTORY.json'
    inv = json.loads(inv_path.read_text())
    sec = CO.resources()
    sec['schema'] = 'MD1-resource-inventory-correction-v1'
    sec['note'] = ('Correction pass only (separate ledger under .private/mechanism_design/correction_v1/compute). The campaign '
                   'figures above are unchanged. Agent (LLM) time is not local CPU; agent shell work is covered by the allowance entry.')
    inv['correction_pass_2026_10_16'] = sec
    inv_path.write_text(json.dumps(inv, indent=1))
    return {k: v for k, v in sec.items() if k != 'units'}


def handoff(extra_path):
    extra = json.loads(Path(extra_path).read_text())
    v1_text = subprocess.run(['git', '-C', str(ROOT), 'show', f'{CORRECTION_BASE}:results/pcrl_mechanism_design_v1/HANDOFF.json'],
                             capture_output=True).stdout
    out = {
        'schema': 'MD1-handoff-v2',
        'written_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'head_at_write': git('rev-parse', 'HEAD'),
        'supersedes': {'handoff_v1_commit': CORRECTION_BASE, 'handoff_v1_sha256': hashlib.sha256(v1_text).hexdigest(),
                       'original_public_commit': 'e141330fbb241c004dd154ec023e714bf56ce401'},
        'campaign_commits': git('log', '--format=%h %cI %s', f'54f089d3e..{CORRECTION_BASE}').splitlines(),
        'correction_commits': git('log', '--format=%h %cI %s', f'{CORRECTION_BASE}..HEAD').splitlines(),
        'artifacts_sha256': {a: sha(RESULTS / a) for a in CO.ARTIFACTS + NEW_ARTIFACTS},
        **extra,
    }
    (RESULTS / 'HANDOFF.json').write_text(json.dumps(out, indent=1))
    return {k: out[k] for k in ('branch', 'head_at_write')}


def main(argv=None):
    argv = argv or sys.argv[1:]
    res = resources() if argv[0] == 'resources' else handoff(argv[1])
    print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
