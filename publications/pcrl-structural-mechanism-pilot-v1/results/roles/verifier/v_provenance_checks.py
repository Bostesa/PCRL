#!/usr/bin/env python3
"""Verifier, independent code: ordering (check 1), proposal integrity (check 2) and no-science-run (check 4).

Read-only git only (log, ls-tree, show, merge-base --is-ancestor). Text files only under .private (ledger.jsonl, the
campaign access-log path, which must be absent) and the source DEV_REPORT_ACCESS.jsonl. Nothing is written except
V_PROVENANCE_CHECKS.json next to this script.

Run from the worktree root:
  <repository>/.venv/bin/python -B results/pcrl_structural_mechanism_pilot_v1/roles/verifier/v_provenance_checks.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BRANCH = 'research/pcrl-structural-mechanism-pilot-v1'
CAMP = 'results/pcrl_structural_mechanism_pilot_v1'
CRITERIA_COMMIT = '2d96c471e'
PROPOSALS = {'designer_1': (f'{CAMP}/roles/designer_1/PROPOSAL.md', '6bc0617b286dccd2e7b428c51b15c6399696cacf5737415188d6db46f0aa4002'),
             'designer_2': (f'{CAMP}/roles/designer_2/PROPOSAL.md', '74b420674660468e4cbf6373b0353611b44da79721fd9abf00eb1591753a2437')}
SRC_ACCESS = Path('<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/access/DEV_REPORT_ACCESS.jsonl')


def git(*args, binary=False):
    r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, check=False)
    return r.stdout if binary else r.stdout.decode()


def main():
    out = {'head': git('rev-parse', 'HEAD').strip(), 'branch_tip': git('rev-parse', BRANCH).strip()}
    # ---- check 1: ordering
    log = git('log', '--format=%H %cI %s', BRANCH, '--', CAMP, 'structural_pilot').splitlines()
    out['campaign_commits'] = [{'commit': l[:40], 'committed': l[41:66], 'subject': l[67:]} for l in log]
    crit = f'{CAMP}/roles/coordinator/SELECTION_CRITERIA.md'
    out['criteria_commits_all_refs'] = git('log', '--all', '--format=%H %cI', '--', crit).split('\n')[:-1]
    out['tree_at_criteria_commit'] = git('ls-tree', '-r', '--name-only', CRITERIA_COMMIT, '--', CAMP).split()
    out['designer_files_at_criteria_commit'] = [p for p in out['tree_at_criteria_commit'] if '/designer_' in p]
    ordering = {}
    for d, (path, _) in PROPOSALS.items():
        add = git('log', '--diff-filter=A', '--format=%H %cI', '--', path).split()
        touching = git('log', '--all', '--format=%H', '--', path).split()
        anc = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', CRITERIA_COMMIT, add[0]]).returncode == 0
        st = os.stat(ROOT / path)
        ordering[d] = {'added_in': add[0], 'added_at': add[1], 'commits_touching_all_refs': touching,
                       'criteria_commit_is_ancestor': anc, 'file_birth_local': st.st_birthtime, 'file_mtime_local': st.st_mtime}
    cst = os.stat(ROOT / crit)
    ordering['criteria_file'] = {'file_birth_local': cst.st_birthtime, 'committed': out['criteria_commits_all_refs']}
    out['ordering'] = ordering
    out['check1_pass'] = bool(len(out['criteria_commits_all_refs']) == 1 and not out['designer_files_at_criteria_commit']
                              and all(ordering[d]['criteria_commit_is_ancestor'] for d in PROPOSALS)
                              and all(ordering[d]['file_birth_local'] > cst.st_birthtime for d in PROPOSALS))
    # ---- check 2: proposal integrity
    integ = {}
    for d, (path, expected) in PROPOSALS.items():
        add = ordering[d]['added_in']
        h = {'worktree': hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
             'HEAD': hashlib.sha256(git('show', f'HEAD:{path}', binary=True)).hexdigest(),
             'adding_commit': hashlib.sha256(git('show', f'{add}:{path}', binary=True)).hexdigest()}
        integ[d] = {'expected': expected, **h, 'all_equal_expected': all(v == expected for v in h.values()),
                    'edited_after_adding': len(ordering[d]['commits_touching_all_refs']) != 1}
    out['integrity'] = integ
    out['check2_pass'] = all(v['all_equal_expected'] and not v['edited_after_adding'] for v in integ.values())
    # ---- check 4: no science run
    ledger = [json.loads(l) for l in (ROOT / '.private/pcrl_structural_mechanism_pilot_v1/compute/ledger.jsonl').read_text().splitlines() if l.strip()]
    units = {}
    for e in ledger:
        if e['event'] in ('unit_start', 'unit_end'):
            u = units.setdefault(e['unit_id'], {})
            if e['event'] == 'unit_start':
                u.update({'stage': e['stage'], 'argv_module': next((a for a in e['argv'] if a.startswith('structural_pilot.')), None)})
            else:
                u.update({'status': e['status'], 'returncode': e['returncode'], 'cpu_s': e['charged_cpu_s']})
    science_modules = {'structural_pilot.fit', 'structural_pilot.readers', 'structural_pilot.report', 'structural_pilot.bank',
                       'structural_pilot.mechanism'}
    out['ledger'] = {'lines': len(ledger), 'events': [e['event'] for e in ledger], 'units': units,
                     'charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in ledger if e['event'] == 'charge'],
                     'science_units': [u for u, v in units.items() if v.get('argv_module') in science_modules or v.get('stage') in ('fit', 'readers', 'report', 'bank')]}
    camp_log = ROOT / '.private/pcrl_structural_mechanism_pilot_v1/access/DEV_REPORT_REREAD.jsonl'
    src_lines = SRC_ACCESS.read_text().splitlines()
    src_events = [json.loads(l) for l in src_lines if l.strip()]
    locks = {n: (ROOT / CAMP / n).exists() for n in ('ANALYSIS_LOCK.json', 'SELECTION_LOCK.json', 'REPORTING_LOCK.json', 'SERVICE_BANK_MANIFEST.json')}
    priv_dirs = {d: (ROOT / '.private/pcrl_structural_mechanism_pilot_v1' / d).exists() for d in ('runs', 'bank', 'access')}
    out['access'] = {'campaign_reread_log_exists': camp_log.exists(), 'source_log_lines': len(src_lines),
                     'source_log_events': [(e.get('event'), e.get('campaign'), e.get('utc') or e.get('recorded_utc')) for e in src_events],
                     'source_log_events_for_this_campaign': [e for e in src_events if e.get('campaign') == 'pcrl_structural_mechanism_pilot_v1'],
                     'source_log_mtime_local': os.stat(SRC_ACCESS).st_mtime, 'campaign_locks_present': locks,
                     'campaign_private_dirs_present': priv_dirs, 'mechanism_module_exists': (ROOT / 'structural_pilot/mechanism.py').exists()}
    out['check4_pass'] = bool(not out['ledger']['science_units'] and not out['access']['campaign_reread_log_exists']
                              and len(src_lines) == 5 and not out['access']['source_log_events_for_this_campaign']
                              and not any(locks.values()) and not any(priv_dirs.values()))
    (HERE / 'V_PROVENANCE_CHECKS.json').write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: out[k] for k in ('head', 'check1_pass', 'check2_pass', 'check4_pass', 'designer_files_at_criteria_commit',
                                          'integrity', 'ledger', 'access')}, indent=1, default=str))


if __name__ == '__main__':
    main()
