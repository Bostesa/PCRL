"""Verifier procedural checks of the REPAIR_FREEZE (PROTOCOL section 8; label-free: records, access log, ledger and git metadata).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
  F1 the freeze is committed before the first repair-panel label read for any new release (check_repair / verify), and no panel
     read other than the registered whole-role preflight operations precedes it (its own count must be 0);
  F2 every R-partition artifact and selection hashed in the freeze is unchanged now (witness, routing, evaluation, directions,
     selections and the selected strengths), and no realized repair outcome predates the freeze;
  F3 the frozen rule / code hashes equal the current files, and the recorded recipe-selection outcome equals RECIPE_SELECTION.json;
  F4 R-partition fits (directions, compression codes) and their label reads relative to the freeze time (fits after the freeze must not
     read a repair panel before their fit, and are listed).

    python3 results/pcrl_correction_transfer_stability_v1/roles/verifier/code/v_freeze.py --tag V1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def t(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(dt.timezone.utc)


def sha_or_none(p):
    p = Path(p)
    return V.file_sha(p) if p.is_file() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    fz_path = RES / 'REPAIR_FREEZE.json'
    fz = json.loads(fz_path.read_text())
    rc, log = git('log', '--diff-filter=A', '--format=%h %cI', '--', str(fz_path))
    first = log.splitlines()[-1].split() if log else None
    rc2, later = git('log', '--format=%h %cI', '--', str(fz_path))
    commit_utc = t(first[1]) if first else None
    access = [json.loads(l) for l in (CAMP / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if l.strip()]
    panel_new = sorted((e for e in access if e['touches_repair_panel'] and e['operation'] not in ('source_replay', 'support_count')),
                       key=lambda e: e['utc'])
    before = [e for e in panel_new if commit_utc is None or t(e['utc']) <= commit_utc]
    out = []
    ok1 = commit_utc is not None and not before and fz.get('repair_panel_label_reads_before_freeze') == 0 and \
        len(later.splitlines()) == 1
    out.append({'id': 'F1_FREEZE_BEFORE_PANEL_READS', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'REPAIR_FREEZE.json committed before the first repair-panel label read for a new release; its own count of earlier '
                        'panel reads (0) reproduced; the freeze file never changed after its commit',
                'method': 'git history of REPAIR_FREEZE.json against LABEL_ACCESS.jsonl (touches_repair_panel, operations other than '
                          'source_replay / support_count)',
                'status': 'PASS' if ok1 else 'FAIL',
                'numbers': {'freeze_commit': first, 'freeze_record_utc': fz['utc'], 'commits_touching_freeze': later.splitlines(),
                            'first_panel_read_after_freeze': (panel_new[0]['utc'], panel_new[0].get('unit_id'), panel_new[0]['operation'])
                            if panel_new else None,
                            'panel_reads_before_freeze_commit': [(e['utc'], e.get('unit_id'), e['operation']) for e in before]}})
    # F2: R artifacts unchanged since the freeze; no realized repair outcome predates it
    changed, sel_changed = [], []
    for P, rec in fz['repair_partitions'].items():
        d = TRACKF / P
        for n, h in rec['witness'].items():
            if sha_or_none(d / 'WITNESS' / n) != h:
                changed.append(f'{P}/WITNESS/{n}')
        for n, h in rec['routing'].items():
            if sha_or_none(d / 'ROUTING' / n) != h:
                changed.append(f'{P}/ROUTING/{n}')
        if sha_or_none(d / 'EVAL' / 'WITNESS_EVAL_MS.json') != rec['witness_eval_ms']:
            changed.append(f'{P}/EVAL/WITNESS_EVAL_MS.json')
        for did, files in rec['directions'].items():
            for n, h in files.items():
                if sha_or_none(d / 'DIR' / did / n) != h:
                    changed.append(f'{P}/DIR/{did}/{n}')
        for stem, s in rec['selections'].items():
            f = d / 'SELECT' / f'{stem}.json'
            if sha_or_none(f) != s['sha256']:
                sel_changed.append(f'{P}/SELECT/{stem}')
            elif json.loads(f.read_text())['selected'] != s['selected']:
                sel_changed.append(f'{P}/SELECT/{stem} (selected)')
    real_early = []
    for P in fz['repair_partitions']:
        for f in (TRACKF / P / 'SELECT').glob('*__REALIZED.json'):
            r = json.loads(f.read_text())
            if t(r['utc']) <= t(fz['utc']):
                real_early.append(f'{P}/{f.name}')
    out.append({'id': 'F2_FROZEN_R_ARTIFACTS_UNCHANGED', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'every R-partition witness, routing, witness evaluation, direction file and GUARD selection hashed in the freeze is '
                        'byte-identical now, the selected strengths are unchanged, and no realized repair outcome predates the freeze',
                'method': 'SHA-256 of the files listed in REPAIR_FREEZE.json repair_partitions; REALIZED record times',
                'status': 'PASS' if not (changed or sel_changed or real_early) else 'FAIL',
                'numbers': {'changed_files': changed, 'changed_selections': sel_changed, 'realized_before_freeze': real_early,
                            'selections_frozen': {P: {k: v['selected'] for k, v in rec['selections'].items()}
                                                  for P, rec in fz['repair_partitions'].items()}}})
    # F3: rules / code hashes and the recipe outcome
    code_diff = [k for k, h in fz['rules'].items()
                 if V.file_sha((RES / 'roles' / 'statistical_reviewer' / k) if k.startswith('RULES') else (ROOT / k)) != h]
    allcode = [k for k, h in fz['code_sha256'].items() if sha_or_none(ROOT / k) != h]
    rs = json.loads((RES / 'diagnosis' / 'RECIPE_SELECTION.json').read_text())
    ok3 = (not code_diff and fz['recipe_selection']['outcome'] == rs['outcome'] and fz['recipe_selection']['selected'] == rs['selected']
           and fz['recipe_selection']['file_sha256'] == V.file_sha(RES / 'diagnosis' / 'RECIPE_SELECTION.json')
           and fz['recipe_selection']['gates_file_sha256'] == V.file_sha(RES / 'diagnosis' / 'RECIPE_GATES.json'))
    out.append({'id': 'F3_FROZEN_RULES_AND_RECIPE', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'frozen rule files and rule / selection / ladder / question A / compression code unchanged since the freeze; '
                        'recorded recipe outcome equals RECIPE_SELECTION.json (and its gates file)',
                'method': 'SHA-256 of the files named in REPAIR_FREEZE.json rules and code_sha256',
                'status': 'PASS' if ok3 else 'FAIL',
                'numbers': {'rule_or_code_files_changed': code_diff, 'campaign_modules_changed_since_freeze': allcode,
                            'recipe_outcome': fz['recipe_selection']['outcome'], 'selected': fz['recipe_selection']['selected']}})
    # F4: R fits relative to the freeze
    fits = []
    for P in fz['repair_partitions']:
        for dd in sorted((TRACKF / P / 'DIR').glob('*')):
            j = dd / 'DIR.json'
            if j.is_file():
                r = json.loads(j.read_text())
                fits.append({'what': f'{P}/DIR/{dd.name}', 'utc': r['utc'], 'after_freeze': t(r['utc']) > t(fz['utc']),
                             'in_freeze': dd.name in fz['repair_partitions'][P]['directions']})
        for cd in sorted((TRACKF / P / 'COMP').glob('COMP__*')):
            js = sorted(cd.glob('*.json'))
            if js:
                r = json.loads(js[0].read_text())
                fits.append({'what': f'{P}/COMP/{cd.name}', 'utc': r.get('utc'), 'after_freeze': bool(r.get('utc')) and t(r['utc']) > t(fz['utc']),
                             'unit': r.get('unit_id')})
    late = [f for f in fits if f['after_freeze']]
    late_panel = []
    for f in late:
        u = f.get('unit')
        if u and any(e.get('unit_id') == u and e['touches_repair_panel'] for e in access):
            late_panel.append(f['what'])
    out.append({'id': 'F4_R_FITS_RELATIVE_TO_FREEZE', 'kind': 'PROCEDURAL_COMPLIANCE',
                'what': 'R-partition fits (directions, compression codes) dated relative to the freeze; a fit after the freeze must not have '
                        'read a repair panel',
                'method': 'record times against REPAIR_FREEZE.json utc; LABEL_ACCESS.jsonl lines of the fitting unit',
                'status': 'FAIL' if late_panel else ('ATTENTION' if [f for f in late if f['what'].split('/')[1] == 'DIR'] else 'PASS'),
                'numbers': {'fits': fits, 'fits_after_freeze': late, 'post_freeze_fits_with_panel_reads': late_panel}})
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['id'] = f"{c['id']}__{a.tag}"
        c['code_sha256'] = code
    V.write_record(f'V_FREEZE__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))
    for c in out:
        if c['status'] != 'PASS':
            print(c['id'], json.dumps(c['numbers'])[:1500])


if __name__ == '__main__':
    main()
