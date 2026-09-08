"""Read-only coalition-study comparisons with explicit recipient and audit scope.

This module reads compact scores/metadata only. It never loads raw people,
prediction arrays, fitted estimators, or a representation-training module.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, SPLITS, MARGINS,
    finite, statistics, coverage_status, combined_coverage, parent_comparison,
    feature_bank_comparison, joint_status, read_json, save_json, save_csv, sha,
)
from scripts.summarize_acs_pca16_init import utility_criteria

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('F_I', 'F_Iplus', 'F_J', 'P_I', 'P_Iplus', 'P_J')
SCOPES = ('standard_independent', 'expanded_independent', 'expanded_catchup')
UTILITY_VIEW = {'income_binary': 'A', 'civilian_at_work': 'A', 'same_residence': 'A',
                'public_coverage': 'B', 'commute_over20': 'B'}
AUDIT_ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
               'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
               'AB': ('SEX', 'RAC1P')}
CONTEXT = ('E_pca', 'PCA16', 'E_pca_leace', 'B_rich_bank', 'C_tree_bank',
           'E_bank', 'S_bank', 'E_F_C', 'E_F_D', 'E_K_C', 'E_K_D',
           'S_F_C', 'S_F_D', 'S_K_C', 'S_K_D')
ROLE_FIELDS = ('seed', 'condition', 'role', 'view', 'target', 'audit_budget')
INDEX_FIELDS = (*ROLE_FIELDS, 'scope')
COMPACT_EXPORTS = ('PER_CLASS.csv', 'PER_CANDIDATE.csv', 'AUDIT_CURVES.csv', 'CANDIDATE_LINEAGE.json')


def key(row, scope=None):
    return tuple(row.get(field) for field in ROLE_FIELDS) + ((scope,) if scope is not None else ())


def build_index(rows):
    """Selections must be explicit; never choose a development extremum."""
    result = {}
    candidates = set()
    for row in rows:
        identity = (*key(row), row['candidate_id'])
        if identity in candidates:
            raise ValueError('Duplicate candidate identity: ' + str(identity))
        candidates.add(identity)
        for scope in row.get('selected_scopes', ()):
            identity = key(row, scope)
            if identity in result:
                raise ValueError('Multiple selected candidates: ' + str(identity))
            result[identity] = row
    return result


def selected(index, seed, condition, role, view, target, budget=None, scope=None):
    scope = scope or ('utility' if role == 'utility' else 'native' if role == 'native' else SCOPES[0])
    return index.get((seed, condition, role, view, target, budget, scope))


def loss(row, split):
    return row.get('scores', {}).get(split, {}).get('log_loss') if row else None


def aggregate_rows(rows, fields, value_field='left_minus_right'):
    groups = defaultdict(dict)
    availability = defaultdict(dict)
    for row in rows:
        identity = tuple(row.get(name) for name in fields)
        if row['seed'] in groups[identity]:
            raise ValueError('Repeated seed in paired aggregate: ' + str(identity))
        groups[identity][row['seed']] = row.get(value_field)
        availability[identity][row['seed']] = row.get('present', row.get(value_field) is not None)
    result = []
    for identity, values in groups.items():
        strict = statistics(values.get(seed) for seed in (0, 1, 2))
        present = [values[seed] for seed in (0, 1, 2) if availability[identity].get(seed, False)]
        observed = statistics(present)
        result.append(dict(zip(fields, identity), **strict,
            observed_mean=observed['mean'], observed_sample_sd=observed['sample_sd'],
            n_present=len(present), present_scores_defined=observed['complete'],
            partial_seed_summary=len(present) < 3))
    return result


def audit_coverage(row, metadata, split, exposed_row, target=None):
    """Binary task audits stay binary; only RAC1P has nine declared classes."""
    if row is None:
        expected = 9 if target == 'RAC1P' else 2
        result = coverage_status(expected, None, None, None, None)
        result['target'] = target
        result['opposing_task_threshold'] = None
        if target in ATTRIBUTES:
            result['limitation_census_codes'] = {k: [i+1 for i in values] for k, values in result['limitations'].items()}
        return result
    expected = 9 if row['target'] == 'RAC1P' else 2
    score = row.get('scores', {}).get(split, {})
    if score.get('n_classes', expected) != expected:
        raise ValueError('Target schema differs from protocol: ' + row['target'])
    control = exposed_row.get('scores', {}).get(split, {}) if exposed_row else {}
    result = coverage_status(expected, metadata.get('fit_support'),
        row.get('scores', {}).get('validation', {}).get('support'), score.get('support'),
        [c.get('recall') for c in control.get('per_class', ())])
    result['target'] = row['target']
    result['candidate_id'] = row['candidate_id']
    result['opposing_task_threshold'] = None if row['target'] not in ATTRIBUTES else 'original descriptive references'
    if row['target'] in ATTRIBUTES:
        result['limitation_census_codes'] = {k: [i+1 for i in values] for k, values in result['limitations'].items()}
    return result


def contrast_matrix():
    result = []
    for interface in ('F', 'P'):
        result.extend((name, interface+'_'+left, interface+'_'+right) for name, left, right in
                      (('J_minus_I', 'J', 'I'), ('J_minus_Iplus', 'J', 'Iplus'), ('Iplus_minus_I', 'Iplus', 'I')))
    result.extend(('F_minus_P', 'F_'+regime, 'P_'+regime) for regime in ('I', 'Iplus', 'J'))
    result.extend(('learned_minus_direct_E', condition, 'E') for condition in CONDITIONS)
    return result


def paired_rows(index, seeds):
    result = []
    for comparison, left, right in contrast_matrix():
        for seed in seeds:
            for role in ('utility', 'native', 'audit'):
                roles = ([(UTILITY_VIEW[t], t) for t in UTILITY_TASKS if role == 'utility' or t in SOURCE_TASKS]
                         if role != 'audit' else [(v, t) for v, targets in AUDIT_ROLES.items() for t in targets])
                if role == 'native' and right == 'E':
                    continue
                for view, target in roles:
                    for budget in ((120, 360) if role == 'audit' else (None,)):
                        for scope in (SCOPES if role == 'audit' else (role,)):
                            a = selected(index, seed, left, role, view, target, budget, scope)
                            b = selected(index, seed, right, role, view, target, budget, scope)
                            for raw_split, split in SPLITS.items():
                                native_split = raw_split.replace('validation', 'source_validation') if role == 'native' else raw_split
                                av, bv = loss(a, native_split), loss(b, native_split)
                                value = av-bv if finite(av) and finite(bv) else None
                                result.append(dict(seed=seed, comparison=comparison, left=left, right=right,
                                    role=role, view=view, target=target, audit_budget=budget, scope=scope,
                                    split=native_split if role == 'native' and raw_split.startswith('validation') else split,
                                    left_loss=av, right_loss=bv, left_minus_right=value,
                                    present=a is not None and b is not None,
                                    signed_gain_left_minus_right=-value if role == 'audit' and value is not None else None))
    return result


def control_row(index, seed, condition, target, budget, scope=SCOPES[0]):
    """Controls use the canonical A role; they do not gain a fictitious observer."""
    for view in ('A', 'control', 'AB', 'B'):
        for candidate_scope in (scope, SCOPES[0], 'control', condition):
            row = selected(index, seed, condition, 'audit', view, target, budget, candidate_scope)
            if row is not None:
                return row
            if condition == 'prior':
                row = selected(index, seed, condition, 'audit', view, target, None, candidate_scope)
                if row is not None:
                    return row
    return None


def per_seed_rows(index, seeds):
    result = []
    for condition in (*CONDITIONS, 'E'):
        for seed in seeds:
            for role in ('utility', 'native', 'audit'):
                if role == 'native' and condition == 'E':
                    continue
                roles = ([(UTILITY_VIEW[t], t) for t in UTILITY_TASKS if role == 'utility' or t in SOURCE_TASKS]
                         if role != 'audit' else [(v, t) for v, ts in AUDIT_ROLES.items() for t in ts])
                for view, target in roles:
                    for budget in ((120, 360) if role == 'audit' else (None,)):
                        for scope in (SCOPES if role == 'audit' else (role,)):
                            row = selected(index, seed, condition, role, view, target, budget, scope)
                            prior = control_row(index, seed, 'prior', target, budget) if role == 'audit' else None
                            for raw_split, split in SPLITS.items():
                                use_split = raw_split.replace('validation', 'source_validation') if role == 'native' else raw_split
                                value, reference = loss(row, use_split), loss(prior, raw_split)
                                result.append(dict(seed=seed, condition=condition, role=role, view=view, target=target,
                                    audit_budget=budget, scope=scope,
                                    split=use_split if role == 'native' and raw_split.startswith('validation') else split, log_loss=value,
                                    prior_loss=reference, signed_prior_relative_gain=reference-value
                                    if finite(reference) and finite(value) else None,
                                    selected_candidate=row['candidate_id'] if row else None,
                                    present=row is not None,
                                    origin_metrics=row.get('origin_metrics') if row else None))
    return result


def aggregate_metrics(rows):
    fields = ('condition', 'role', 'view', 'target', 'audit_budget', 'scope', 'split')
    result = []
    for metric in ('log_loss', 'signed_prior_relative_gain'):
        source = rows if metric == 'log_loss' else [r for r in rows if r['role'] == 'audit']
        result.extend({'metric': metric, **row} for row in aggregate_rows(source, fields, metric))
    return result


def budget_rows(index, seeds):
    rows = []
    for seed in seeds:
        for condition in (*CONDITIONS, 'E'):
            for view, targets in AUDIT_ROLES.items():
                for target in targets:
                    for scope in SCOPES:
                        a = selected(index, seed, condition, 'audit', view, target, 120, scope)
                        b = selected(index, seed, condition, 'audit', view, target, 360, scope)
                        for raw_split, split in SPLITS.items():
                            av, bv = loss(a, raw_split), loss(b, raw_split)
                            rows.append(dict(seed=seed, condition=condition, view=view, target=target, scope=scope,
                                split=split, loss120=av, loss360=bv,
                                loss360_minus120=bv-av if finite(av) and finite(bv) else None,
                                candidate120=a['candidate_id'] if a else None, candidate360=b['candidate_id'] if b else None))
    return rows


def scope_rows(index, seeds):
    """Report access expansions separately from longer fitting."""
    rows = []
    for seed in seeds:
        for condition in (*CONDITIONS, 'E'):
            for view, targets in AUDIT_ROLES.items():
                for target in targets:
                    for budget in (120, 360):
                        for left, right in ((SCOPES[1], SCOPES[0]), (SCOPES[2], SCOPES[1])):
                            a = selected(index, seed, condition, 'audit', view, target, budget, left)
                            b = selected(index, seed, condition, 'audit', view, target, budget, right)
                            for raw_split, split in SPLITS.items():
                                av, bv = loss(a, raw_split), loss(b, raw_split)
                                rows.append(dict(seed=seed, condition=condition, view=view, target=target, audit_budget=budget,
                                    left_scope=left, right_scope=right, split=split,
                                    left_minus_right=bv-av if finite(av) and finite(bv) else None,
                                    metric='signed_gain_left_minus_right', left_candidate=a['candidate_id'] if a else None,
                                    right_candidate=b['candidate_id'] if b else None))
    return rows


def coalition_singleton_rows(index, seeds):
    rows = []
    for seed in seeds:
        for condition in (*CONDITIONS, 'E'):
            for target in ATTRIBUTES:
                for budget in (120, 360):
                    for scope in SCOPES:
                        coalition = selected(index, seed, condition, 'audit', 'AB', target, budget, scope)
                        for singleton in ('A', 'B'):
                            individual = selected(index, seed, condition, 'audit', singleton, target, budget, scope)
                            for raw_split, split in SPLITS.items():
                                av, bv = loss(coalition, raw_split), loss(individual, raw_split)
                                if raw_split == 'validation' and finite(av) and finite(bv) and av > bv+1e-12:
                                    raise ValueError('Coalition selector is worse than an included singleton on validation')
                                rows.append(dict(seed=seed, condition=condition, target=target, audit_budget=budget,
                                    scope=scope, singleton=singleton, split=split, coalition_loss=av, singleton_loss=bv,
                                    coalition_minus_singleton_loss=av-bv if finite(av) and finite(bv) else None,
                                    coalition_minus_singleton_gain=bv-av if finite(av) and finite(bv) else None,
                                    development_monotonicity_required=False))
    return rows


def mean_sd(record):
    if not record or not record['complete']:
        if record and record.get('present_scores_defined') and record.get('n_present', 0):
            sd = f" ± {record['observed_sample_sd']:.6f}" if record['observed_sample_sd'] is not None else ''
            return f"{record['observed_mean']:.6f}{sd} ({record['n_present']}/3; partial)"
        return 'unavailable' if not record else f"undefined ({record['n_defined']}/3)"
    return f"{record['mean']:.6f} ± {record['sample_sd']:.6f}"


def markdown_table(headers, rows):
    def line(values):
        return '| '+' | '.join(str(x).replace('|', '/') for x in values)+' |'
    return '\n'.join((line(headers), line(['---']*len(headers)), *(line(row) for row in rows)))


class Evidence:
    def __init__(self):
        self.rows = []
        self.inputs = set()
        self.completed_seeds = []
        self.audit_manifests = {}

    def read(self, path):
        path = Path(path).resolve()
        self.inputs.add(path)
        return read_json(path)

    def add(self, raw, metadata, origin):
        row = copy.deepcopy(raw)
        row['metadata'] = copy.deepcopy(metadata)
        row['origin_metrics'] = str(Path(origin).relative_to(ROOT))
        self.rows.append(row)


def load_evidence(out):
    evidence = Evidence()
    config = evidence.read(out/'config.json')
    for name in ('PROTOCOL.md', 'PURPOSE_POLICY.md', 'ACCESS_SCOPE.md'):
        evidence.inputs.add(out/name)
    for name in ('EXECUTION_AMENDMENTS.json', 'EXECUTED_MATRIX.json', 'FITTING_COUNTS.json', 'progress.json', 'FIRST_BLOCK_PROJECTION.json'):
        if (out/name).exists():
            evidence.inputs.add(out/name)
    historical_controls = {}
    for seed in config['seeds']:
        directory = out/f'seed_{seed}'
        if not (directory/'complete.json').exists():
            continue
        complete = evidence.read(directory/'complete.json')
        if complete['seed'] != seed or not complete['six_pairs_complete']:
            raise ValueError('Completed seed identity differs from path')
        for relative, expected in complete['compact_sha256'].items():
            if sha(directory/relative) != expected:
                raise ValueError('Completed evidence changed: '+str(directory/relative))
        evidence.completed_seeds.append(seed)
        for condition in (*CONDITIONS, 'E'):
            base = directory/condition
            path = base/'metrics.json'
            record = evidence.read(path)
            selection = evidence.read(base/'selection_before_test.json')
            audits = evidence.read(base/'audits/audit_selection.json')
            evidence.audit_manifests[seed, condition] = audits
            for raw in record['raw_metrics']:
                if raw['seed'] != seed or raw['condition'] != condition:
                    raise ValueError('Candidate identity differs from its condition path')
                role = raw['view']+'/'+raw['target']
                if raw['role'] == 'audit':
                    meta = audits['candidates'][str(raw['audit_budget'])][role][raw['candidate_id']]
                    expected = [scope for scope, cid in audits['selections'][str(raw['audit_budget'])][role].items()
                                if cid == raw['candidate_id']]
                    if set(expected) != set(raw['selected_scopes']):
                        raise ValueError('Scored selection differs from pre-test manifest')
                else:
                    meta = selection['utility_metadata'][role]['candidates'][raw['candidate_id']]
                    if raw['view'] != UTILITY_VIEW[raw['target']]:
                        raise ValueError('Utility head used an unauthorized recipient view')
                evidence.add(raw, meta, path)
        control_path = directory/'controls/metrics.json'
        controls = evidence.read(control_path)
        meta = evidence.read(directory/'controls/selection_before_test.json')
        for raw in controls['raw_metrics']:
            if raw['condition'] == 'prior':
                for budget in (120, 360):
                    item = {**raw, 'audit_budget': budget, 'actual_fitting_budget': None}
                    evidence.add(item, meta['prior_metadata'][raw['target']], control_path)
            else:
                target, budget = raw['target'], raw['audit_budget']
                source = meta['metadata'][target]
                if source.get('reused'):
                    if seed not in historical_controls:
                        historical_controls[seed] = evidence.read(ROOT/config['preservation_reference_results']/'extended'/f'seed_{seed}'/'selection_before_test.json')
                    cm = historical_controls[seed]['budgets'][str(budget)]['fitting_records'][f'audit/exposed/{target}']['candidates'][raw['candidate_id']]
                else:
                    cm = source[f'nested{budget}']['candidates'][raw['candidate_id']]
                evidence.add(raw, cm, control_path)
        native_path = directory/'native_source.json'
        native = evidence.read(native_path)
        grouped = {}
        for raw in native['raw_metrics']:
            identity = raw['condition'], raw['view'], raw['target']
            row = grouped.setdefault(identity, dict(seed=seed, condition=raw['condition'], role='native',
                view=raw['view'], target=raw['target'], audit_budget=None, candidate_id='native',
                selected_scopes=['native'], scores={}, prediction_sha256={}))
            row['scores'][raw['split']] = raw['score']
            row['scores'][raw['split']+'_person_weighted'] = raw['person_weighted']
            row['prediction_sha256'][raw['split']] = raw['prediction_sha256']
        for raw in grouped.values():
            evidence.add(raw, {'family': 'fixed_native_head', 'selection': 'fixed final source head'}, native_path)
    return evidence, config


def export_candidates(out, evidence):
    scores, classes, fits, curves, lineage = [], [], [], [], []
    curve_seen = set()
    for raw in evidence.rows:
        meta = raw['metadata']
        common = {k: raw.get(k) for k in (*ROLE_FIELDS, 'candidate_id', 'selected_scopes', 'origin_metrics')}
        common.update(family=meta.get('family'), candidate_origin=meta.get('candidate_origin'),
                      space=meta.get('space'), source_view=meta.get('source_view'),
                      source_candidate_id=meta.get('source_candidate_id'), diagnostic_only=meta.get('diagnostic_only', False))
        for split, score in raw['scores'].items():
            display_split = SPLITS.get(split, split)
            scores.append({**common, 'split': display_split, **{k: v for k, v in score.items() if k != 'per_class'}})
            for cls in score.get('per_class', ()):
                index = cls['class_index']
                support = meta.get('fit_support') or ()
                classes.append({**common, 'split': display_split, **cls,
                    'fit_class_support': support[index] if len(support) > index else None,
                    'original_census_code': index+1 if raw['role'] == 'audit' and raw['target'] in ATTRIBUTES else None})
        accounting = {k: meta.get(k) for k in ('selected_epoch', 'seed', 'input_dim', 'actual_auditor_input_dimension',
            'fit_rows', 'fit_support', 'validation_hashes', 'fit_hashes', 'parameters', 'optimizer_steps',
            'training_row_exposures', 'row_exposure_min', 'row_exposure_max', 'schedule_seed', 'schedule_hash',
            'fallback_reason', 'fit_warnings', 'inherited_exposure', 'base_candidate_directory', 'base_metadata_sha256')}
        accounting['initialization_seed'] = accounting.pop('seed')
        fits.append({**common, **accounting})
        # Candidate fitting seeds must not overwrite the experimental-unit seed.
        lineage.append({**{k: v for k, v in meta.items()
                           if k not in ('seed', 'validation_curve', 'validation_scores', 'parameters')},
                        'candidate_seed': meta.get('seed'), **common})
        # An inherited candidate is one existing fit. Keep its candidate link above,
        # but emit a learning curve only once per actual candidate/budget checkpoint.
        origin = meta.get('base_candidate_directory') or (raw['origin_metrics'], raw['view'], raw['target'], raw['candidate_id'])
        curve_id = (str(origin), raw['audit_budget'])
        if curve_id not in curve_seen:
            curve_seen.add(curve_id)
            curves.extend({**common, **point, 'selected_checkpoint': point.get('epoch') == meta.get('selected_epoch')}
                          for point in meta.get('validation_curve', ()))
    for name, rows in (('PER_CANDIDATE.csv', scores), ('PER_CLASS.csv', classes),
                       ('FITTING.csv', fits), ('AUDIT_CURVES.csv', curves)):
        save_csv(out/name, rows)
    save_json(out/'CANDIDATE_LINEAGE.json', lineage)
    return {'score_rows': len(scores), 'class_rows': len(classes), 'candidate_records': len(fits), 'curve_rows': len(curves)}


def compact_exports(out):
    """Deterministic standalone gzip files; plain derived exports remain local."""
    result = {}
    for name in COMPACT_EXPORTS:
        source = out/name
        target = out/(name+'.gz')
        stream = io.BytesIO()
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=9, mtime=0) as archive:
            archive.write(source.read_bytes())
        target.write_bytes(stream.getvalue())
        result[name] = {'plain_sha256': sha(source), 'gzip_sha256': sha(target),
            'plain_bytes': source.stat().st_size, 'gzip_bytes': target.stat().st_size,
            'published_file': target.name, 'plain_file_local_only': True}
    save_json(out/'COMPACT_EXPORTS.json', {'format': 'gzip, compression level9, mtime0, empty original filename',
        'decompress': "gzip.decompress(Path('PER_CLASS.csv.gz').read_bytes())", 'files': result})
    return result


def load_context(config, evidence):
    from scripts import summarize_acs_restricted as restricted
    from scripts import summarize_acs_selective as shared
    history, _, _ = restricted.load_evidence(ROOT/config['restricted_reference_results'])
    evidence.inputs.update(history.inputs)
    return history, {b: shared.index_records(history.records(b)) for b in (120, 360)}


def old_selected(indices, seed, release, role, target, budget=360, scope='primary'):
    return indices[budget].get((seed, role, release, target, scope))


def old_loss(indices, seed, release, role, target, split, budget=360, scope='primary'):
    row = old_selected(indices, seed, release, role, target, budget, scope)
    return row.get(split, {}).get('log_loss') if row else None


def contextual_rows(index, indices, seeds):
    references, comparisons = [], []
    for seed in seeds:
        for release in CONTEXT:
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                for target in targets:
                    for budget in ((120, 360) if role == 'audit' else (None,)):
                        for original_scope in (('primary', 'pooled') if role == 'audit' else ('primary',)):
                            old = old_selected(indices, seed, release, role, target, budget or 120, original_scope)
                            for raw_split, split in SPLITS.items():
                                value = old.get(raw_split, {}).get('log_loss') if old else None
                                references.append(dict(seed=seed, historical_release=release, role=role, target=target,
                                    report_audit_budget=budget, actual_audit_budget=old.get('evidence_audit_budget') if old else None,
                                    original_scope=original_scope, split=split, log_loss=value,
                                    original_candidate=old['candidate_id'] if old else None,
                                    origin_metrics=old.get('origin_metrics') if old else None,
                                    causal_replacement=False, respects_new_individual_policy='not established'))
                                if original_scope != 'primary':
                                    continue
                                for condition in CONDITIONS:
                                    scopes = SCOPES if role == 'audit' else ('utility',)
                                    for scope in scopes:
                                        view = 'AB' if role == 'audit' else UTILITY_VIEW[target]
                                        row = selected(index, seed, condition, 'audit' if role == 'audit' else 'utility', view, target, budget, scope)
                                        learned = loss(row, raw_split)
                                        comparisons.append(dict(seed=seed, left=condition, right=release,
                                            comparison='contextual_learned_minus_historical', role='audit' if role == 'audit' else 'utility',
                                            view=view, target=target, audit_budget=budget, scope=scope,
                                            historical_scope='primary', historical_actual_audit_budget=old.get('evidence_audit_budget') if old else None,
                                            split=split, left_loss=learned, right_loss=value,
                                            left_minus_right=learned-value if finite(learned) and finite(value) else None,
                                            signed_gain_left_minus_right=value-learned if role == 'audit' and finite(learned) and finite(value) else None,
                                            causal_replacement=False))
    return references, comparisons


def contextual_report(out, references):
    fields = ('historical_release', 'role', 'target', 'report_audit_budget', 'actual_audit_budget',
              'original_scope', 'split', 'causal_replacement', 'respects_new_individual_policy')
    aggregate = aggregate_rows(references, fields, 'log_loss')
    save_csv(out/'CONTEXTUAL_AGGREGATE.csv', aggregate)
    shown = ('E_pca', 'PCA16', 'E_pca_leace', 'B_rich_bank', 'C_tree_bank', 'E_bank', 'S_bank')
    names = {'E_pca': 'Original PCA32', 'PCA16': 'Original PCA16', 'E_pca_leace': 'PCA32 + LEACE',
             'B_rich_bank': 'Rich neural bank', 'C_tree_bank': 'Rich tree bank',
             'E_bank': 'E source-only bank', 'S_bank': 'Sham source-only bank'}
    def row_for(release, target, split):
        role = 'audit' if target in ATTRIBUTES else 'transfer'
        return next((r for r in aggregate if r['historical_release'] == release and r['target'] == target
                     and r['split'] == split and r['role'] == role and r['original_scope'] == 'primary'
                     and r['report_audit_budget'] == (360 if role == 'audit' else None)), None)
    lines = ['# Contextual reference losses', '',
        'These historical releases do not implement the new A/B policy and cannot replace its matched F/P comparisons. '
        'Rich and source-only banks differ in purpose routing, source-loss weights, interfaces and inherited exposure; '
        'the new objective uses source weights .25/.25/.5. Full original source identities remain in '
        '[CONTEXTUAL.csv](CONTEXTUAL.csv). Original PCA32 is the fixed parent for existing margins.', '',
        'Each cell is mean ± sample SD over three cohort-sharing seeds. U and PWGTP score the same original '
        'validation-selected predictions. Utility losses are lower-is-better; **attacker log losses are higher when '
        'that selected predictor recovers less**. These are raw losses, not the gains in the primary decision table. '
        'Attack rows use historical independent selections; their actual budget is shown explicitly. '
        'No new fits, fresh holdout, matched two-purpose routing, or complete race support is implied.', '']
    for split in ('development_evaluation', 'development_evaluation_person_weighted'):
        lines.extend(['## '+('PWGTP' if split.endswith('weighted') else 'Unweighted')+' development evaluation', '',
            markdown_table(['Reference', 'Income', 'Employment', 'Coverage', 'Residence', 'Commute'],
                [[names[c], *[mean_sd(row_for(c, t, split)) for t in
                   ('income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')]] for c in shown]), '',
            markdown_table(['Reference', 'SEX attack log loss', 'RAC1P attack log loss', 'Actual epochs SEX / RAC1P'],
                [[names[c], *[mean_sd(row_for(c, t, split)) for t in ATTRIBUTES],
                  ' / '.join(str((row_for(c, t, split) or {}).get('actual_audit_budget', 'unavailable')) for t in ATTRIBUTES)] for c in shown]), ''])
    lines.extend(['[All raw reference aggregates](CONTEXTUAL_AGGREGATE.csv) retain additional single-interface arms and original audit scopes. '
        '[Every contextual paired seed contrast](CONTEXTUAL_PAIRED.csv) and '
        '[paired means/SDs](CONTEXTUAL_PAIRED_AGGREGATE.csv) preserve the actual historical budget and scope. '
        'They are descriptive comparisons, not causal replacements or statistical noninferiority tests.', ''])
    (out/'CONTEXTUAL_REFERENCES.md').write_text('\n'.join(lines))
    return aggregate


def criterion_rows(index, history, old_indices, seeds):
    utility, policies, feature, support = [], [], [], []
    old_meta = {b: history.selections(b) for b in (120, 360)}
    from scripts.summarize_acs_protection import audit_coverage as old_coverage
    for seed in seeds:
        for raw_split, split in SPLITS.items():
            parent = {t: old_loss(old_indices, seed, 'E_pca', 'transfer', t, raw_split, 120) for t in UTILITY_TASKS}
            bank_res = {r: old_loss(old_indices, seed, r, 'transfer', 'same_residence', raw_split, 120)
                        for r in ('B_rich_bank', 'C_tree_bank')}
            utilities = {c: {t: loss(selected(index, seed, c, 'utility', UTILITY_VIEW[t], t), raw_split)
                              for t in UTILITY_TASKS} for c in (*CONDITIONS, 'E')}
            for condition, scores in utilities.items():
                utility.append(dict(seed=seed, condition=condition, split=split,
                                    **utility_criteria(scores, parent, bank_res)))
            for budget in (120, 360):
                priors = {t: loss(control_row(index, seed, 'prior', t, budget), raw_split) for t in ATTRIBUTES}
                parent_attack = {t: old_loss(old_indices, seed, 'E_pca', 'audit', t, raw_split, budget) for t in ATTRIBUTES}
                parent_cov = {t: old_coverage(old_indices[budget], old_meta[budget][seed], seed, 'E_pca', t,
                                             raw_split, 'primary') for t in ATTRIBUTES}
                for scope in SCOPES:
                    covered = {}
                    attacked = {}
                    for condition in (*CONDITIONS, 'E'):
                        for view, targets in AUDIT_ROLES.items():
                            for target in targets:
                                row = selected(index, seed, condition, 'audit', view, target, budget, scope)
                                exposed = control_row(index, seed, 'exposed', target, budget)
                                cover = audit_coverage(row, row.get('metadata', {}) if row else {}, raw_split, exposed, target)
                                covered[condition, view, target] = cover
                                support.append({'seed': seed, 'condition': condition, 'view': view, 'target': target,
                                    'audit_budget': budget, 'scope': scope, 'split': split, **cover})
                            attacked[condition, view] = {t: loss(selected(index, seed, condition, 'audit', view, t, budget, scope), raw_split)
                                                         for t in ATTRIBUTES}
                            cov = {t: combined_coverage(covered[condition, view, t], parent_cov[t]) for t in ATTRIBUTES}
                            policies.append(dict(seed=seed, condition=condition, view=view, audit_budget=budget,
                                scope=scope, split=split, fixed_parent='E_pca', parent_actual_audit_budget=120,
                                opposing_target_assessment='continuous scores only; no declared pass threshold',
                                **parent_comparison(parent, utilities[condition], bank_res, parent_attack,
                                                    attacked[condition, view], priors, cov)))
                    for task in ('same_residence', 'commute_over20'):
                        recipient = UTILITY_VIEW[task]
                        for comparator in ('P_J', 'E'):
                            for access in (recipient, 'AB'):
                                result = feature_bank_comparison(utilities['F_J'][task], utilities[comparator][task],
                                    attacked['F_J', access], attacked[comparator, access], priors,
                                    {t: combined_coverage(covered['F_J', access, t], covered[comparator, access, t]) for t in ATTRIBUTES})
                                feature.append(dict(seed=seed, task=task, authorized_view=recipient, sensitive_access=access,
                                    comparator=comparator, audit_budget=budget, scope=scope, split=split,
                                    numeric_joint_inequalities=joint_status([result['utility']['pass'],
                                        *[value['numeric_inequality'] for value in result['attributes'].values()]]),
                                    separate_reserved_task_decision=True, **result))
    return utility, policies, feature, support


def stage_native_rows(evidence):
    rows = []
    for raw in evidence.rows:
        if raw['role'] != 'native':
            continue
        for split, score in raw['scores'].items():
            rows.append(dict(seed=raw['seed'], condition=raw['condition'], view=raw['view'], target=raw['target'],
                             split=SPLITS.get(split, split), log_loss=score['log_loss']))
    return rows, aggregate_rows(rows, ('condition', 'view', 'target', 'split'), 'log_loss')


def gradient_rows(out, evidence):
    """Flatten scalar diagnostics while keeping their exact frozen point names."""
    rows = []
    for seed in evidence.completed_seeds:
        directory = out/f'seed_{seed}'/'training'
        for path in sorted(directory.glob('*.json')):
            record = evidence.read(path)
            def visit(value, trail):
                if isinstance(value, dict):
                    if 'groups' in value and 'coefficients' in value and isinstance(value['groups'], dict):
                        for group, values in value['groups'].items():
                            rows.append(dict(seed=seed, diagnostic_path='/'.join(trail), group=group,
                                source_record=str(path.relative_to(ROOT)), interface=value.get('interface'),
                                condition=value.get('condition'), coefficients=value['coefficients'],
                                losses=value.get('losses'), state_unchanged=value.get('state_unchanged'),
                                rng_unchanged=value.get('rng_unchanged'), **values))
                        return
                    for name, child in value.items():
                        visit(child, (*trail, name))
                elif isinstance(value, list):
                    for i, child in enumerate(value):
                        if isinstance(child, (dict, list)):
                            visit(child, (*trail, str(i)))
            visit(record, ())
    save_csv(out/'GRADIENT_DIAGNOSTICS.csv', rows)
    return rows


def table_report(out, aggregate, native_aggregate, utility, complete):
    lookup = {(r['condition'], r['role'], r['view'], r['target'], r['audit_budget'], r['scope'], r['split'], r['metric']): r for r in aggregate}
    def cell(condition, target, split, scope=SCOPES[0], view=None):
        role = 'audit' if target in ATTRIBUTES else 'utility'
        view = view or ('AB' if role == 'audit' else UTILITY_VIEW[target])
        metric = 'signed_prior_relative_gain' if role == 'audit' else 'log_loss'
        return mean_sd(lookup.get((condition, role, view, target, 360 if role == 'audit' else None,
                                  scope if role == 'audit' else 'utility', split, metric)))
    lines = ['# Two-purpose development comparison', '',
        'Complete three-seed matrix.' if complete else '**PARTIAL MATRIX: missing seeds remain undefined in means.**', '',
        'All losses/gains are nats. Lower utility loss and lower signed prior-relative attack gain are better. '
        'Mean ± sample SD describes three fits on the same cohort; partial summaries explicitly show available n/3. '
        'They are not population uncertainty. '
        'PWGTP scores the same unweighted-validation-selected predictions. Test households remain DEVELOPMENT EVALUATION.', '',
        'F receives 16 features per purpose; P receives two A source probabilities and one B probability. '
        'Iplus adds sensitive-only local pressure; J adds coalition pressure. Read [policy](PURPOSE_POLICY.md) '
        'and [access scopes](ACCESS_SCOPE.md) before interpreting J−I or a pooled score.', '',
        '[Every selected seed/role](PER_SEED.csv), [all candidate scores](PER_CANDIDATE.csv.gz), '
        '[all classes](PER_CLASS.csv.gz), [fixed paired contrasts](PAIRED.csv), [lineage/exposure](CANDIDATE_LINEAGE.json.gz). '
        'The four large derived exports use deterministic gzip; [format and hashes](COMPACT_EXPORTS.json).', '']
    for split in ('development_evaluation', 'development_evaluation_person_weighted'):
        lines.extend(['## '+split.replace('_', ' '), '',
            markdown_table(['Condition', 'Income A', 'Employment A', 'Coverage B', 'Residence A', 'Commute B', 'All-source margins /3'],
                [[c, *[cell(c, t, split) for t in ('income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')],
                  sum(r['source_preservation']['pass'] is True for r in utility if r['condition'] == c and r['split'] == split)]
                 for c in (*CONDITIONS, 'E')]), ''])
        for scope in SCOPES:
            lines.extend([f'### Coalition sensitive recovery: {scope}, 360 epochs', '',
                markdown_table(['Condition', 'AB SEX gain', 'AB RAC1P gain'],
                               [[c, cell(c, 'SEX', split, scope), cell(c, 'RAC1P', split, scope)] for c in (*CONDITIONS, 'E')]), ''])
        lines.extend(['### Individual forbidden recovery, expanded catch-up scope', '',
            'Opposing task gains have no predeclared pass threshold. Reserved opposing targets have no catch-up. '
            'E has no saved observer; its expanded-catchup selector contains the available independent candidates only.', ''])
        for view, targets in AUDIT_ROLES.items():
            if view == 'AB':
                continue
            rows = []
            for condition in (*CONDITIONS, 'E'):
                cells = [mean_sd(lookup.get((condition, 'audit', view, t, 360, SCOPES[2], split, 'signed_prior_relative_gain')))
                         for t in targets]
                rows.append([condition, *cells])
            lines.extend([markdown_table(['Recipient '+view, *targets], rows), ''])
        lines.extend(['### Fixed native source heads', '',
            'These heads receive representation-fitting source supervision. Independent utility probes above use their separate 2,048-label budget. No better-of selection is made.', '',
            markdown_table(['Condition', 'Income A', 'Employment A', 'Coverage B'],
                [[c, *[mean_sd(next((r for r in native_aggregate if r['condition'] == c and r['target'] == t and r['split'] == split), None))
                       for t in SOURCE_TASKS]] for c in CONDITIONS]), ''])
    lines.extend(['## Context and limitations', '',
        'Original PCA32 remains the parent for source and residential-retention denominators. Its unchanged historical audit is120 epochs, '
        'so parent-halving references are not advertised as a matched360 parent audit. Historical PCA16/banks and prior single-interface arms '
        'remain [contextual reference tables](CONTEXTUAL_REFERENCES.md), not causal replacements for a new paired system. '
        'Historical rich and current source-only banks differ in purpose routing, source-loss weights '
        '(this study uses .25/.25/.5), interfaces and inherited exposure; they cannot replace the matched F/P comparisons. '
        'All nine race categories remain reported; missing independent attacker-fitting/validation support leaves full race protection unassessable. '
        'Saved observers inherit a separate representation-fitting exposure pool, whose support is recorded independently. '
        'Independent failures and expanded candidate minima do not certify privacy.', '',
        'The [feature capability criteria](feature_capability.json) evaluate residence and commute separately against P-J and direct E, '
        'with own-recipient and coalition sensitive-gain references. [All original-parent references](criteria.json) retain failed and undefined margins.'])
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')


def plots(out, aggregate, native_aggregate, budget, scope_changes, pair_aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    import numpy as np
    def get(c, target, split, role='utility', view=None, scope=SCOPES[0], metric='log_loss'):
        view = view or ('AB' if role == 'audit' else UTILITY_VIEW[target])
        row = next((r for r in aggregate if r['condition'] == c and r['target'] == target and r['role'] == role
                    and r['view'] == view and r['split'] == split and r['scope'] == (scope if role == 'audit' else role)
                    and r['metric'] == metric and (role != 'audit' or r['audit_budget'] == 360)), None)
        return (row['mean'] if row['mean'] is not None else row.get('observed_mean', np.nan)) if row else np.nan
    colors = {'F': 'tab:blue', 'P': 'tab:orange', 'E': 'gray'}
    marks = {'I': 'o', 'Iplus': 's', 'J': '*'}
    n = max((r.get('n_present', 0) for r in aggregate), default=0)
    caption = f'Development evaluation; {n}/3 seed means' + (' (partial)' if n < 3 else '')
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex='row', sharey='row')
        for i, split in enumerate(('development_evaluation', 'development_evaluation_person_weighted')):
            for j, scope in enumerate(SCOPES):
                ax = axes[i, j]
                for c in (*CONDITIONS, 'E'):
                    ax.scatter(get(c, 'same_residence', split), get(c, target, split, 'audit', scope=scope, metric='signed_prior_relative_gain'),
                        color=colors[c[0]], marker=marks[c.split('_')[1]] if '_' in c else 'D', s=90 if c.endswith('_J') else 45, label=c)
                ax.set(xlabel='A residence log loss (nats)', ylabel=f'AB {target} gain (nats)', title=('Unweighted' if i == 0 else 'PWGTP')+'; '+scope.replace('_', ' '))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=9)
        fig.suptitle(f'{caption}\nCoalition {target}: 360-epoch validation-selected finite audits; residence shown, commute assessed separately')
        fig.tight_layout(rect=(0, .06, 1, .93)); fig.savefig(out/f'coalition_{target}.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, task in zip(axes, ('same_residence', 'commute_over20')):
        positions = np.arange(7)
        for i, split in enumerate(('development_evaluation', 'development_evaluation_person_weighted')):
            values = [get(c, task, split) for c in (*CONDITIONS, 'E')]
            ax.plot(positions, values, 'o-' if i == 0 else 's--', label='Unweighted' if i == 0 else 'PWGTP')
        ax.set_xticks(positions, (*CONDITIONS, 'E'), rotation=45); ax.set(title=task, ylabel='Authorized-view log loss (nats)'); ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(caption+'\nReserved tasks: separate capability questions, same selected predictions')
    fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(out/'reserved_task_capability.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, task in zip(axes, SOURCE_TASKS):
        for condition in CONDITIONS:
            stages = ('I', 'W', condition)
            values = [next((r['mean'] if r['mean'] is not None else r.get('observed_mean') for r in native_aggregate if r['condition'] == stage and r['target'] == task
                            and r['split'] == 'development_evaluation'), np.nan) for stage in stages]
            ax.plot(range(3), values, marker=marks[condition.split('_')[1]], color=colors[condition[0]],
                    linestyle='-' if condition[0] == 'F' else '--', label=condition)
        ax.set_xticks(range(3), ('I', 'Common W', 'Final')); ax.set(title=task, ylabel='Native source log loss (nats)'); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8)
    fig.suptitle(caption+'\nUnweighted native source heads: parallel continuations; no reserved-task stage probes')
    fig.tight_layout(rect=(0,.09,1,.92)); fig.savefig(out/'native_source_stages.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, target in enumerate(ATTRIBUTES):
        for j, split in enumerate(('development_evaluation', 'development_evaluation_person_weighted')):
            ax = axes[i, j]
            for transition, marker in (((SCOPES[1], SCOPES[0]), 'o'), ((SCOPES[2], SCOPES[1]), 's')):
                values = []
                for condition in CONDITIONS:
                    rr = [r for r in scope_changes if r['condition'] == condition and r['view'] == 'AB'
                          and r['target'] == target and r['audit_budget'] == 360 and r['split'] == split
                          and (r['left_scope'], r['right_scope']) == transition]
                    values.append(statistics(r['left_minus_right'] for r in rr)['mean'])
                ax.plot(range(6), values, marker+'-', label='Public head additions' if transition[0] == SCOPES[1] else 'Saved-start additions')
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set_xticks(range(6), CONDITIONS, rotation=45); ax.set(title=target+'; '+('PWGTP' if j else 'Unweighted'), ylabel='Change in selected AB attack gain (nats)')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(caption+'\nExpanding legal attack access: development and validation changes can differ')
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(out/'audit_access_expansion.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, target in zip(axes, ATTRIBUTES):
        for scope in SCOPES:
            values = []
            for condition in CONDITIONS:
                rr = [r for r in budget if r['condition'] == condition and r['view'] == 'AB' and r['target'] == target
                      and r['scope'] == scope and r['split'] == 'development_evaluation']
                values.append(statistics(-r['loss360_minus120'] if r['loss360_minus120'] is not None else None for r in rr)['mean'])
            ax.plot(range(6), values, 'o-', label=scope.replace('_', ' '))
        ax.axhline(0, color='gray', linewidth=.7); ax.set_xticks(range(6), CONDITIONS, rotation=45)
        ax.set(title=target, ylabel='AB gain360 − gain120, unweighted (nats)'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(caption+'\nNested audit-budget sensitivity; all individual roles and weights remain in CSV')
    fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(out/'audit_budget.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, target in enumerate(ATTRIBUTES):
        for j, split in enumerate(('development_evaluation', 'development_evaluation_person_weighted')):
            ax = axes[i, j]
            for view, marker in (('A', 'o'), ('B', 's'), ('AB', 'D')):
                values = [get(c, target, split, 'audit', view=view, scope=SCOPES[2],
                              metric='signed_prior_relative_gain') for c in (*CONDITIONS, 'E')]
                ax.plot(range(7), values, marker+'-', label=view)
            ax.set_xticks(range(7), (*CONDITIONS, 'E'), rotation=45)
            ax.set(title=target+'; '+('PWGTP' if j else 'Unweighted'), ylabel='Signed prior-relative recovery gain (nats)')
            ax.grid(alpha=.2); ax.legend(title='Recipient', fontsize=8)
    fig.suptitle(caption+'\nExpanded-catchup360: individual and coalition selections; no development monotonicity imposed')
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(out/'individual_coalition_recovery.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    task_order = ('income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')
    task_labels = ('Income A', 'Employment A', 'Coverage B', 'Residence A', 'Commute B')
    for j, split in enumerate(('development_evaluation', 'development_evaluation_person_weighted')):
        ax = axes[j]
        for offset, regime in zip((-.12, 0, .12), ('I', 'Iplus', 'J')):
            records = [next(r for r in pair_aggregate if r['comparison'] == 'F_minus_P'
                       and r['left'] == 'F_'+regime and r['role'] == 'utility'
                       and r['target'] == task and r['split'] == split) for task in task_order]
            values = [r['mean'] if r['mean'] is not None else r.get('observed_mean') for r in records]
            deviations = [r['sample_sd'] if r['sample_sd'] is not None else 0 for r in records]
            ax.errorbar(np.arange(5)+offset, values, yerr=deviations, marker='o', linestyle='-', capsize=3, label=regime)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xticks(range(5), task_labels, rotation=30); ax.set(title='PWGTP' if j else 'Unweighted', ylabel='F loss − P loss, nats')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(caption+'\nPaired task differences: negative favors features; bars are descriptive seed SDs')
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(out/'task_F_minus_P.png', dpi=170); plt.close(fig)


def analysis_report(out, aggregate, pairs, pair_aggregate, feature, support, budgets, scopes, complete):
    def contrast(comparison, left, right, target, split, role='utility', view=None, scope=SCOPES[2]):
        view = view or ('AB' if role == 'audit' else UTILITY_VIEW[target])
        row = next((r for r in pair_aggregate if r['comparison'] == comparison and r['left'] == left and r['right'] == right
                    and r['target'] == target and r['split'] == split and r['role'] == role and r['view'] == view
                    and r['scope'] == (scope if role == 'audit' else role)
                    and r['audit_budget'] == (360 if role == 'audit' else None)), None)
        if role == 'audit' and row:
            row = {**row, 'mean': -row['mean'] if row['mean'] is not None else None,
                   'observed_mean': -row['observed_mean'] if row.get('observed_mean') is not None else None}
        return mean_sd(row)
    def both(comparison, left, right, target, role='utility'):
        return ' / '.join(contrast(comparison, left, right, target, split, role) for split in
                          ('development_evaluation', 'development_evaluation_person_weighted'))
    lines = ['# Fixed coalition contrasts', '',
        'All six conditions and three seeds are complete.' if complete else '**PARTIAL: incomplete seeds are not averaged away.**', '',
        'The comparisons below were declared before fitting. They are descriptive differences on reused development households, '
        'not significance tests or privacy guarantees. Utility differences are left loss minus right loss; negative is better. '
        'Attack differences are left signed gain minus right signed gain; negative means less measured recovery. '
        'Cells show mean ± sample SD, unweighted / PWGTP, on the same selected predictions.', '',
        '## Coalition objective and stronger local pressure', '',
        'J−I and J−Iplus answer different questions. Iplus changes only the additional local SEX/RAC1P terms. '
        'Nominal per-target/branch coefficients match J, while realized gradients and observer information need not. '
        'Opposing-source gradients are unchanged. [Frozen policy](PURPOSE_POLICY.md).', '',
        markdown_table(['Contrast', 'A residence loss Δ U / W', 'B commute loss Δ U / W', 'AB SEX gain Δ U / W', 'AB race gain Δ U / W'],
            [[f'{left} − {right}', both(name, left, right, 'same_residence'), both(name, left, right, 'commute_over20'),
              both(name, left, right, 'SEX', 'audit'), both(name, left, right, 'RAC1P', 'audit')]
             for name, left, right in contrast_matrix() if name in ('J_minus_I', 'J_minus_Iplus', 'Iplus_minus_I')]), '',
        'Attack cells above use expanded_catchup360. [Every paired role/scope/budget/seed](PAIRED.csv) and '
        '[aggregates](PAIRED_AGGREGATE.csv) retain standard independent and public-head-expanded conclusions separately. '
        'A gain that exists only before inherited/public/saved-start candidates enter is not robust to that legal access.', '',
        '## Feature capability is task-specific', '',
        'Residence belongs to A and commute to B. The .01-nat feature-advantage reference is evaluated separately. '
        'The .005-nat sensitive-gain excess references are evaluated for the task recipient and AB. '
        'Neither a residence result nor a global conjunction substitutes for a commute result.', '',
        markdown_table(['Comparison', 'Residence loss Δ U / W', 'Commute loss Δ U / W', 'AB SEX gain Δ U / W', 'AB race gain Δ U / W'],
            [[f'{left} − {right}', both(name, left, right, 'same_residence'), both(name, left, right, 'commute_over20'),
              both(name, left, right, 'SEX', 'audit'), both(name, left, right, 'RAC1P', 'audit')]
             for name, left, right in contrast_matrix() if name == 'F_minus_P' or left == 'F_J' and right == 'E']), '']
    margin_rows = []
    for split in ('development_evaluation', 'development_evaluation_person_weighted'):
        for task in ('same_residence', 'commute_over20'):
            for comparator in ('P_J', 'E'):
                rr = [r for r in feature if r['task'] == task and r['comparator'] == comparator and r['split'] == split
                      and r['audit_budget'] == 360 and r['scope'] == SCOPES[2] and r['sensitive_access'] == 'AB']
                margin_rows.append(['W' if split.endswith('weighted') else 'U', task,
                    comparator, sum(r['utility']['pass'] is True for r in rr),
                    sum(r['numeric_joint_inequalities'] is True for r in rr),
                    sum(r['coverage_complete'] for r in rr)])
    lines.extend([markdown_table(['Scoring', 'Reserved task', 'F-J comparator', '.01 utility inequality /3',
                                 'Utility + both AB gain inequalities /3', 'Complete sensitive support /3'], margin_rows), '',
        'All task-specific own-view and coalition checks, including numerical versus support-qualified status, are in '
        '[feature_capability.json](feature_capability.json). Opposing-task recovery has no invented pass threshold.', '',
        '## Audit scope, budget and evidence limits', '',
        'AB includes every legal singleton sensitive candidate, not only each singleton winner. Its selected validation loss '
        'cannot exceed any included candidate on the same rows. The validation winner can have worse development loss than '
        'a singleton development winner. [Coalition-minus-singleton evidence](COALITION_MINUS_SINGLETON.csv) retains that distinction.', '',
        'Public native heads add fit budget and a composed function family for F. P reuses the identical wire candidates; '
        'its hidden features remain unavailable. [Scope expansions](AUDIT_SCOPE.csv), [budget differences](AUDIT_BUDGET.csv), '
        '[candidate lineage](CANDIDATE_LINEAGE.json.gz), [all learning curves](AUDIT_CURVES.csv.gz) and [support](SUPPORT.csv) expose these differences.', ''])
    lines.extend(['The standalone saved-observer candidate is diagnostic and excluded from selection. Epoch zero remains '
        'eligible inside its catch-up trajectory, so an expanded-catchup advantage can reflect inherited fitting exposure '
        'without beneficial additional optimization. [Audit construction and exposure](COALITION_AUDIT.md).', ''])
    changed = [r for r in budgets if r['loss360_minus120'] is not None and r['loss360_minus120'] != 0]
    cases = sorted({(r['seed'], r['condition'], r['view'], r['target'], r['scope']) for r in changed})
    lines.extend([f'Nested120/360 selected losses differ in {len(cases)} seed/condition/view/target/scope cases '
                  f'({len(changed)} scoring cells). A zero change means the selected trajectory checkpoint stayed competitive; '
                  'it does not establish a complete attacker class or privacy.', '',
        'RAC1P retains the nine-class schema. Missing independent attacker-fitting/validation support and exposed-control failures remain visible '
        'rather than repaired by changing the population. Full race and all-target protection are not certified. '
        'Representation-fitting observer support is a separate exposure pool. '
        'The deterministic repeated-output check duplicates the same version; it is not the fresh-noise experiment in the historical exact note.', '',
        'The final [research decision](RESEARCH_DECISION.md) interprets coalition benefit over I, survival against Iplus, '
        'residence and commute capability, source retention and audit reversals as separate findings.'])
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')


def summarize(out):
    out = Path(out).resolve()
    evidence, config = load_evidence(out)
    index = build_index(evidence.rows)
    history, context_indices = load_context(config, evidence)
    per_seed = per_seed_rows(index, config['seeds'])
    aggregate = aggregate_metrics(per_seed)
    pairs = paired_rows(index, config['seeds'])
    fields = ('comparison', 'left', 'right', 'role', 'view', 'target', 'audit_budget', 'scope', 'split')
    pair_aggregate = aggregate_rows(pairs, fields)
    for row in pair_aggregate:
        row['signed_gain_difference_mean'] = -row['mean'] if row['role'] == 'audit' and row['mean'] is not None else None
        row['signed_gain_difference_observed_mean'] = -row['observed_mean'] if row['role'] == 'audit' and row['observed_mean'] is not None else None
    utility, policies, feature, support = criterion_rows(index, history, context_indices, config['seeds'])
    budgets, scope_changes = budget_rows(index, config['seeds']), scope_rows(index, config['seeds'])
    singletons = coalition_singleton_rows(index, config['seeds'])
    contextual, contextual_pairs = contextual_rows(index, context_indices, config['seeds'])
    context_fields = ('comparison', 'left', 'right', 'role', 'view', 'target', 'audit_budget', 'scope',
                      'historical_scope', 'historical_actual_audit_budget', 'split', 'causal_replacement')
    contextual_pair_aggregate = aggregate_rows(contextual_pairs, context_fields)
    for row in contextual_pair_aggregate:
        row['signed_gain_difference_mean'] = -row['mean'] if row['role'] == 'audit' and row['mean'] is not None else None
    contextual_report(out, contextual)
    native, native_aggregate = stage_native_rows(evidence)
    counts = export_candidates(out, evidence)
    compressed = compact_exports(out)
    gradients = gradient_rows(out, evidence)
    exports = [('PER_SEED.csv', per_seed), ('AGGREGATE.csv', aggregate), ('PAIRED.csv', pairs),
        ('PAIRED_AGGREGATE.csv', pair_aggregate), ('SUPPORT.csv', support), ('AUDIT_BUDGET.csv', budgets),
        ('AUDIT_SCOPE.csv', scope_changes), ('COALITION_MINUS_SINGLETON.csv', singletons),
        ('CONTEXTUAL.csv', contextual), ('CONTEXTUAL_PAIRED.csv', contextual_pairs),
        ('CONTEXTUAL_PAIRED_AGGREGATE.csv', contextual_pair_aggregate),
        ('NATIVE_SOURCE.csv', native), ('NATIVE_AGGREGATE.csv', native_aggregate)]
    for name, rows in exports:
        save_csv(out/name, rows)
    save_json(out/'criteria.json', {'margins': MARGINS, 'fixed_parent': 'E_pca', 'parent_actual_audit_budget': 120,
        'per_seed_split_utility': utility, 'per_seed_view_budget_scope_policy': policies,
        'opposing_task_threshold': None, 'all_target_certificate': False})
    save_json(out/'feature_capability.json', {'margins': MARGINS, 'tasks_decided_separately': True, 'per_seed': feature})
    complete = evidence.completed_seeds == config['seeds']
    table_report(out, aggregate, native_aggregate, utility, complete)
    analysis_report(out, aggregate, pairs, pair_aggregate, feature, support, budgets, scope_changes, complete)
    plots(out, aggregate, native_aggregate, budgets, scope_changes, pair_aggregate)
    result = {'evaluation_status': config['evaluation_status'], 'complete': complete,
        'completed_seeds': evidence.completed_seeds, 'final_paired_systems': 6*len(evidence.completed_seeds),
        'aggregate_metrics': aggregate, 'paired_aggregate_metrics': pair_aggregate,
        'native_aggregate_metrics': native_aggregate, 'row_counts': counts,
        'compact_exports': compressed,
        'gradient_diagnostic_rows': len(gradients),
        'report_source_sha256': sha(Path(__file__)),
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(evidence.inputs)}}
    save_json(out/'summary.json', result)
    print({k: result[k] for k in ('complete', 'completed_seeds', 'final_paired_systems', 'row_counts')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)
