#!/usr/bin/env python3
"""Read-only, stdlib review of frozen source-guard audit choices and budget changes.

This complements model/score replay. It never loads fitted models or raw data,
and does not make new audit selections: it checks the recorded validation minima.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('standard_independent', 'expanded_independent', 'expanded_catchup')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def selected_identity(candidate_id, metadata):
    return candidate_id, metadata.get('selected_state_hash'), metadata.get('selected_epoch')


def review(out):
    tick = time.perf_counter()
    manifest = read(out/'RELEASE_MANIFEST.json')
    assert manifest['all_intended_releases_frozen'] and len(manifest['systems']) == 48
    inputs = {str((out/'RELEASE_MANIFEST.json').relative_to(ROOT)): sha(out/'RELEASE_MANIFEST.json')}
    rows, changes, epoch_zero, shifts = [], [], [], []
    counts = defaultdict(Counter)
    checks = 0
    for entry in manifest['systems']:
        dest = ROOT/entry['evidence_directory']
        completion = dest/'complete.json'
        if not completion.exists():
            assert entry['reused'] and dest.parent.parent.name == 'redesign_20260908_acs_coalition_v1'
            completion = dest.parent/'complete.json'  # Historical coalition completed a whole seed.
        for path in (completion, dest/'metrics.json', dest/'audits/audit_selection.json'):
            inputs[str(path.relative_to(ROOT))] = sha(path)
        audit = read(dest/'audits/audit_selection.json')
        metrics = read(dest/'metrics.json')['raw_metrics']
        scored = {(str(r['audit_budget']), r['view']+'/'+r['target'], r['candidate_id']): r
                  for r in metrics if r['role'] == 'audit'}
        groups = ('all', 'reused' if entry['reused'] else 'new',
                  ('reused_' if entry['reused'] else 'new_')+entry['interface'])
        for role in audit['selections']['360']:
            chosen = {}
            for budget in ('120', '360'):
                for scope in SCOPES:
                    cid = audit['selections'][budget][role][scope]
                    pool = audit['selection_pools'][budget][role][scope]
                    assert 'saved_adversary' not in cid
                    candidate = audit['candidates'][budget][role][cid]
                    value = scored[budget, role, cid]
                    ll = value['scores']['validation']['log_loss']
                    assert ll == min(scored[budget, role, c]['scores']['validation']['log_loss'] for c in pool)
                    assert scope in value['selected_scopes']
                    checks += 1
                    chosen[budget, scope] = (cid, candidate, value)
                    row = dict(seed=entry['seed'], condition=entry['condition'],
                               original_condition=entry['original_condition'], reused=entry['reused'],
                               interface=entry['interface'], role=role, budget=int(budget), scope=scope,
                               candidate_id=cid, origin=candidate.get('candidate_origin'),
                               selected_epoch=candidate.get('selected_epoch'),
                               selected_state_hash=candidate.get('selected_state_hash'),
                               catchup=candidate.get('audit_kind') == 'catchup',
                               validation_log_loss=ll,
                               development_log_loss=value['scores']['test']['log_loss'],
                               development_person_weighted_log_loss=value['scores']['test_person_weighted']['log_loss'])
                    rows.append(row)
                    if budget == '360' and scope == 'expanded_catchup':
                        for group in groups:
                            counts[group]['roles'] += 1
                            counts[group]['catchup_eligible_roles'] += any('catchup' in c for c in pool)
                            counts[group]['selected_catchup'] += row['catchup']
                            counts[group]['selected_epoch_zero_catchup'] += row['catchup'] and row['selected_epoch'] == 0
                        if row['catchup'] and row['selected_epoch'] == 0:
                            epoch_zero.append(row)
            for scope in SCOPES:
                c120, m120, r120 = chosen['120', scope]
                c360, m360, r360 = chosen['360', scope]
                assert r360['scores']['validation']['log_loss'] <= r120['scores']['validation']['log_loss']
                changed = selected_identity(c120, m120) != selected_identity(c360, m360)
                for group in groups:
                    counts[group][scope+'_budget_changed'] += changed
                if changed:
                    change = dict(seed=entry['seed'], condition=entry['condition'], reused=entry['reused'],
                                  interface=entry['interface'], role=role, scope=scope,
                                  candidate120=c120, candidate360=c360,
                                  epoch120=m120.get('selected_epoch'), epoch360=m360.get('selected_epoch'))
                    for score in ('validation', 'test', 'test_person_weighted'):
                        change[score+'_360_minus_120'] = r360['scores'][score]['log_loss']-r120['scores'][score]['log_loss']
                    changes.append(change)
            for before, after in zip(SCOPES, SCOPES[1:]):
                old, _, rold = chosen['360', before]
                new, _, rnew = chosen['360', after]
                assert rnew['scores']['validation']['log_loss'] <= rold['scores']['validation']['log_loss']
                if old != new:
                    shifts.append(dict(seed=entry['seed'], condition=entry['condition'], reused=entry['reused'],
                                       interface=entry['interface'], role=role, before=before, after=after,
                                       candidate_before=old, candidate_after=new,
                                       development_delta=rnew['scores']['test']['log_loss']-rold['scores']['test']['log_loss'],
                                       development_person_weighted_delta=rnew['scores']['test_person_weighted']['log_loss']-rold['scores']['test_person_weighted']['log_loss']))
    assert checks == 48*11*2*3
    assert all(sha(ROOT/path) == value for path, value in inputs.items())
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                method='stdlib-only validation-minimum, lineage and budget comparison of immutable evidence; no fitting, model inference, or new selections',
                source_sha256=sha(Path(__file__)), input_files_sha256=inputs,
                units=48, validation_minimum_checks=checks, counts=dict(counts),
                budget_changes=changes, epoch_zero_catchup_winners=epoch_zero,
                scope_changes_360=shifts, selected_rows=rows,
                runtime_seconds=time.perf_counter()-tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/redesign_20260909_acs_source_guard_v1')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), 'Preserve completed review evidence'
    result = review(args.out.resolve())
    with args.report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'units', 'counts', 'runtime_seconds')}, indent=2))


if __name__ == '__main__':
    main()
