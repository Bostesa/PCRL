"""REPAIR_FREEZE.json: everything the repair comparison depends on, frozen before either repair CHECK panel (C0, C1) is opened
(PROTOCOL.md section 8; RULES.md sections 8-9). Label-free: it hashes saved records; it reads no label.

    python3 -m cts.freeze write --plan PLAN.json        writes results/<campaign>/REPAIR_FREEZE.json (write-once)

PLAN.json (coordinator): {"comparison_plan": {...}, "redesign": null | {...}, "notes": [...]}.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import common as C
from . import trackf as TF


def _sha_or_none(p):
    p = Path(p)
    return C.sha(p) if p.is_file() else None


def write(plan_path):
    out = C.RESULTS / C.FREEZE_FILE
    if out.exists():
        raise PermissionError('REPAIR_FREEZE.json exists (write-once)')
    plan = json.loads(Path(plan_path).read_text())
    if not (C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').is_file():
        raise PermissionError('no diagnosis/RECIPE_SELECTION.json: the freeze must record the selected recipe (or NO_RECIPE)')
    head = subprocess.run(['git', '-C', str(C.ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    rparts = {}
    for P in ('R0', 'R1'):
        d = TF.pdir(P)
        rec = {'witness': {n: _sha_or_none(d / 'WITNESS' / n) for n in ('WITNESS.json', 'WITNESS.npz', 'OOF.npz', 'MODEL.pkl')},
               'routing': {n: _sha_or_none(d / 'ROUTING' / n) for n in ('ROUTING.json', 'TEACHER_occupation.json')},
               'witness_eval_ms': _sha_or_none(d / 'EVAL' / 'WITNESS_EVAL_MS.json'), 'directions': {}, 'selections': {}}
        for dd in sorted((d / 'DIR').iterdir()) if (d / 'DIR').is_dir() else []:
            rec['directions'][dd.name] = {n: _sha_or_none(dd / n) for n in ('DIR.json', 'TRAIN.npz', 'DELTA.npz')}
        for sf in sorted((d / 'SELECT').glob('*.json')) if (d / 'SELECT').is_dir() else []:
            if '__REALIZED' in sf.name:
                raise PermissionError(f'{P}: a realized repair outcome exists before the freeze ({sf.name})')
            s = json.loads(sf.read_text())
            rec['selections'][sf.stem] = {'sha256': C.sha(sf), 'selected': s['selected']}
        if rec['witness']['WITNESS.json'] is None or not rec['directions'] or not rec['selections']:
            raise PermissionError(f'{P}: witness, directions and GUARD selections must exist before the freeze')
        rparts[P] = rec
    panel_reads = [e for e in C.access_events() if e.get('touches_repair_panel') and e.get('operation') not in ('source_replay', 'support_count')]
    rec = {'schema': 'CTS1-repair-freeze-v1', 'utc': C.now(), 'research_head_at_freeze': head,
           'rules': {'RULES.json': C.sha(C.RESULTS / 'roles/statistical_reviewer/RULES.json'),
                     'RULES.md': C.sha(C.RESULTS / 'roles/statistical_reviewer/RULES.md'), 'cts/rules.py': C.sha(C.ROOT / 'cts/rules.py'),
                     'cts/select.py': C.sha(C.ROOT / 'cts/select.py'), 'cts/questiona.py': C.sha(C.ROOT / 'cts/questiona.py'),
                     'cts/trackf.py': C.sha(C.ROOT / 'cts/trackf.py'), 'cts/compression.py': C.sha(C.ROOT / 'cts/compression.py')},
           'code_sha256': {str(p.relative_to(C.ROOT)): C.sha(p) for p in sorted((C.ROOT / 'cts').glob('*.py'))},
           'frozen_rules': ['OLD-B80', 'OLD-B50', 'SIMPLE', 'UNCERTAINTY'], 'deployments': ['full', 'ens'],
           'recipe_selection': {'file_sha256': _sha_or_none(C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json'),
                                'gates_file_sha256': _sha_or_none(C.RESULTS / 'diagnosis' / 'RECIPE_GATES.json'),
                                'outcome': (json.loads((C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').read_text())
                                            if (C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').is_file() else {}).get('outcome'),
                                'selected': (json.loads((C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').read_text())
                                             if (C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json').is_file() else {}).get('selected')},
           'redesign': plan.get('redesign'), 'comparison_plan': plan['comparison_plan'], 'notes': plan.get('notes', []),
           'repair_partitions': rparts,
           'repair_panel_label_reads_before_freeze': len(panel_reads),
           'statement': ('frozen before either repair CHECK panel (C0, C1) is opened for any new release; after a repair CHECK outcome is '
                         'seen, revisions using it are adaptive and descriptive and cannot replace or relabel the primary repair result')}
    if panel_reads:
        raise PermissionError(f'repair-panel label reads exist before the freeze: {panel_reads[:3]}')
    C.write_json_x(out, rec)
    print(json.dumps({'written': str(out.relative_to(C.ROOT)), 'recipe': rec['recipe_selection'], 'panel_reads_before': len(panel_reads)}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('write'); x.add_argument('--plan', required=True)
    a = ap.parse_args(argv)
    write(a.plan)
    return 0


if __name__ == '__main__':
    sys.exit(main())
