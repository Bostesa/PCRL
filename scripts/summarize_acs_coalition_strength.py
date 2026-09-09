"""Read-only, fixed-grid analysis of coalition versus local protection strength."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import summarize_acs_coalition as historical
from scripts.acs_coalition_strength_comparisons import (
    SOURCE_TASKS, UTILITY_TASKS, AUDIT_ROLES, finite, difference, complete_and,
    evaluate_pair, fixed_seed_summary, source_check, pareto_membership,
)

ROOT = Path(__file__).resolve().parents[1]
SCOPES = historical.SCOPES
UTILITY_VIEW = historical.UTILITY_VIEW
read_json, save_json, save_csv, sha = historical.read_json, historical.save_json, historical.save_csv, historical.sha
PAIR_FIELDS = ('interface', 'J_beta', 'Iplus_beta', 'J_condition', 'Iplus_condition',
               'attribute', 'weighting', 'scope', 'audit_budget', 'panel', 'delta')
COMPRESSED_EXPORTS = (*historical.COMPACT_EXPORTS, 'UTILITY_MATCHES.csv', 'UTILITY_MATCHES_AGGREGATE.csv',
    'NONDOMINATED_POINTS.csv', 'criteria.json', 'feature_capability.json', 'FITTING.csv',
    'CONTEXTUAL_PAIRED.csv', 'CONTEXTUAL_PAIRED_AGGREGATE.csv', 'PAIRED.csv', 'PAIRED_AGGREGATE.csv', 'SUPPORT.csv')


def beta_key(value):
    if float(value) not in (0., .025, .05, .1, .2):
        raise ValueError('Coefficient outside the frozen grid')
    return format(float(value), '.15g')


def registry(manifest):
    entries = manifest['systems']
    if len(entries) != 54 or len({(e['seed'], e['condition']) for e in entries}) != 54:
        raise ValueError('The physical matrix must have 54 unique paired systems')
    aliases = {}
    for e in entries:
        families = ('J', 'Iplus') if e['beta'] == 0 else (e['family'],)
        for family in families:
            key = (e['seed'], e['interface'], family, beta_key(e['beta']))
            if key in aliases:
                raise ValueError('Repeated fixed coefficient identity')
            aliases[key] = e
    expected = set(itertools.product((0, 1, 2), ('F', 'P'), ('J', 'Iplus'), ('0', '.025', '.05', '.1', '.2')))
    expected = {(s, v, f, beta_key(b)) for s, v, f, b in expected}
    if set(aliases) != expected:
        raise ValueError('Missing or additional family/strength/seed identity')
    return entries, aliases


def checked_read(evidence, path, hashes):
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    if relative not in hashes or sha(path) != hashes[relative]:
        raise ValueError('Compact evidence differs from completion manifest: '+relative)
    return evidence.read(path)


def native_rows(record, original_condition=None, canonical_condition=None):
    grouped = {}
    for raw in record['raw_metrics']:
        if original_condition is not None and raw['condition'] != original_condition:
            continue
        condition = canonical_condition or raw['condition']
        key = raw['seed'], condition, raw['view'], raw['target']
        row = grouped.setdefault(key, dict(seed=raw['seed'], condition=condition, role='native',
            view=raw['view'], target=raw['target'], audit_budget=None, candidate_id='native',
            selected_scopes=['native'], scores={}, prediction_sha256={}))
        for split, score in ((raw['split'], raw['score']), (raw['split']+'_person_weighted', raw['person_weighted'])):
            if split in row['scores']:
                raise ValueError('Duplicated native source score')
            row['scores'][split] = score
        row['prediction_sha256'][raw['split']] = raw['prediction_sha256']
    return list(grouped.values())


def load_evidence(out):
    evidence = historical.Evidence()
    config = evidence.read(out/'config.json')
    rules = evidence.read(out/'comparison_rules.json')
    manifest = evidence.read(out/'REUSE_MANIFEST.json')
    freeze = evidence.read(out/'protocol_freeze.json')
    rule_path = str((out/'comparison_rules.json').relative_to(ROOT))
    if sha(out/'comparison_rules.json') != freeze['scientific_and_protocol_sha256'][rule_path]:
        raise ValueError('Frozen comparison rules changed')
    entries, aliases = registry(manifest)
    old, old_config = historical.load_evidence(ROOT/config['coalition_reference_results'])
    evidence.inputs.update(old.inputs)
    rename = {(e['seed'], e['original_condition']): e['condition'] for e in entries if e['reused']}
    for raw in old.rows:
        row = copy.deepcopy(raw)
        if (row['seed'], row['condition']) in rename:
            row['original_condition'] = row['condition']
            row['condition'] = rename[row['seed'], row['condition']]
            row['reused'] = True
        elif row['condition'] not in ('E', 'prior', 'exposed', 'I', 'W'):
            raise ValueError('Unmapped historical condition')
        evidence.rows.append(row)
    completed = {(e['seed'], e['condition']) for e in entries if e['reused']}
    for e in entries:
        if e['reused']:
            continue
        base = ROOT/e['evidence_directory']
        if not (base/'complete.json').exists():
            continue
        complete = evidence.read(base/'complete.json')
        if complete['seed'] != e['seed'] or complete['condition'] != e['condition'] or not complete['evaluation_complete']:
            raise ValueError('Completed condition identity mismatch')
        release_manifest = out/'RELEASE_MANIFEST.json'
        if not release_manifest.exists() or sha(release_manifest) != complete['global_release_manifest_sha256']:
            raise ValueError('New evaluation lacks its complete global release freeze')
        evidence.inputs.add(release_manifest)
        hashes = complete['files_sha256']
        record = checked_read(evidence, base/'metrics.json', hashes)
        selection = checked_read(evidence, base/'selection_before_test.json', hashes)
        audits = checked_read(evidence, base/'audits/audit_selection.json', hashes)
        for raw in record['raw_metrics']:
            if raw['seed'] != e['seed'] or raw['condition'] != e['condition']:
                raise ValueError('Candidate has wrong seed or condition')
            role = raw['view']+'/'+raw['target']
            if raw['role'] == 'audit':
                metadata = audits['candidates'][str(raw['audit_budget'])][role][raw['candidate_id']]
                expected = [scope for scope, cid in audits['selections'][str(raw['audit_budget'])][role].items() if cid == raw['candidate_id']]
                if set(expected) != set(raw['selected_scopes']):
                    raise ValueError('Audit selection changed after validation freeze')
            elif raw['role'] == 'utility':
                if raw['view'] != UTILITY_VIEW[raw['target']]:
                    raise ValueError('Unauthorized utility view')
                metadata = selection['utility_metadata'][role]['candidates'][raw['candidate_id']]
                expected = selection['utility'][role] == raw['candidate_id']
                if bool(raw['selected_scopes']) != expected:
                    raise ValueError('Utility selection changed after validation freeze')
            else:
                raise ValueError('Unexpected raw result role')
            evidence.add({**raw, 'reused': False}, metadata, base/'metrics.json')
        native = checked_read(evidence, base/'native_source.json', hashes)
        for raw in native_rows(native):
            evidence.add(raw, {'family': 'fixed_native_head'}, base/'native_source.json')
        completed.add((e['seed'], e['condition']))
    # Bind the actual original selected PCA32 heads, not a renamed downstream copy.
    for seed, identity in rules['original_parent_metric_identity'].items():
        for label in ('metrics', 'selection'):
            path = ROOT/identity[label+'_path']
            if sha(path) != identity[label+'_sha256']:
                raise ValueError('Original PCA32 parent evidence changed')
            evidence.inputs.add(path)
    for name in ('PROTOCOL.md', 'PREFIT_REVIEW.md', 'EXECUTED_MATRIX.json', 'FITTING_COUNTS.json'):
        if (out/name).exists():
            evidence.inputs.add(out/name)
    evidence.completed_systems = completed
    return evidence, config, rules, entries, aliases, old_config


def selected(index, seed, condition, role, view, target, budget=None, scope=None):
    return historical.selected(index, seed, condition, role, view, target, budget, scope)


def points_from_index(index, entries, rules):
    points = {}
    references = [dict(seed=s, condition='E', interface='E', family='direct_teacher', beta=None, reused=True) for s in rules['seeds']]
    for e in [*entries, *references]:
        for weighting, split in rules['weights'].items():
            utility = {t: historical.loss(selected(index, e['seed'], e['condition'], 'utility', UTILITY_VIEW[t], t), split) for t in UTILITY_TASKS}
            for scope, budget in itertools.product(rules['audit_scopes'], rules['audit_budgets']):
                gains, losses, candidate_ids, coverage = {}, {}, {}, {}
                for view, targets in AUDIT_ROLES.items():
                    for target in targets:
                        key = view+'/'+target
                        row = selected(index, e['seed'], e['condition'], 'audit', view, target, budget, scope)
                        prior = historical.control_row(index, e['seed'], 'prior', target, budget)
                        losses[key] = historical.loss(row, split)
                        gains[key] = difference(historical.loss(prior, split), losses[key])
                        candidate_ids[key] = row['candidate_id'] if row else None
                        exposed = historical.control_row(index, e['seed'], 'exposed', target, budget)
                        coverage[key] = historical.audit_coverage(row, row.get('metadata', {}) if row else {}, split, exposed, target)
                parent = {t: rules['original_parent_metric_identity'][str(e['seed'])]['tasks'][t]['scores'][split] for t in SOURCE_TASKS}
                key = (e['seed'], e['condition'], weighting, scope, budget)
                points[key] = {**{k: e[k] for k in ('seed', 'condition', 'interface', 'family', 'beta', 'reused')},
                    'weighting': weighting, 'scope': scope, 'audit_budget': budget,
                    'utility': utility, 'gains': gains, 'attack_losses': losses, 'candidate_ids': candidate_ids,
                    'coverage': coverage, 'source_feasibility': source_check({'utility': utility}, parent,
                        roundoff=rules['comparison_roundoff_tolerance']), 'parent': parent}
    return points


def pair_cube(points, aliases, rules):
    rows = []
    for seed, interface, jbeta, lbeta, weighting, scope, budget, panel, delta, attribute in itertools.product(
            rules['seeds'], rules['interfaces'], rules['beta_decimal_strings'], rules['beta_decimal_strings'],
            rules['weights'], rules['audit_scopes'], rules['audit_budgets'], rules['panels'], rules['delta_values'], ('SEX', 'RAC1P')):
        je, le = aliases[seed, interface, 'J', jbeta], aliases[seed, interface, 'Iplus', lbeta]
        j = points[seed, je['condition'], weighting, scope, budget]
        local = points[seed, le['condition'], weighting, scope, budget]
        row = evaluate_pair(j, local, j['parent'], rules['panels'][panel], delta, attribute,
                            rules['comparison_roundoff_tolerance'])
        rows.append(dict(seed=seed, interface=interface, J_beta=jbeta, Iplus_beta=lbeta,
            J_condition=je['condition'], Iplus_condition=le['condition'], attribute=attribute,
            weighting=weighting, scope=scope, audit_budget=budget, panel=panel, delta=delta,
            shared_physical_anchor=je['condition'] == le['condition'],
            J_selected_candidate=j['candidate_ids']['AB/'+attribute], Iplus_selected_candidate=local['candidate_ids']['AB/'+attribute],
            J_attribute_support_complete=j['coverage']['AB/'+attribute]['complete'],
            Iplus_attribute_support_complete=local['coverage']['AB/'+attribute]['complete'],
            attribute_full_schema_assessment='covered' if j['coverage']['AB/'+attribute]['complete'] and local['coverage']['AB/'+attribute]['complete'] else 'unassessable',
            qualification_scope='aggregate numerical inequalities; not an attribute-support certificate', **row))
    if len(rows) != rules['all_pair_grid']['rows_per_attribute']*2:
        raise ValueError('Incomplete frozen comparison cube')
    return rows


def fixed_contrasts(index, entries, aliases, rules):
    contrasts = []
    for seed, interface, beta in itertools.product(rules['seeds'], rules['interfaces'], rules['beta_decimal_strings']):
        j = aliases[seed, interface, 'J', beta]['condition'];local = aliases[seed, interface, 'Iplus', beta]['condition'];anchor = aliases[seed, interface, 'J', '0']['condition']
        contrasts.extend((seed, name, left, right, interface, family, beta) for name, left, right, family in
            [('J_minus_Iplus', j, local, 'paired'), ('J_minus_I', j, anchor, 'J'), ('Iplus_minus_I', local, anchor, 'Iplus')])
    for seed, family, beta in itertools.product(rules['seeds'], rules['families'], rules['beta_decimal_strings']):
        contrasts.append((seed, 'F_minus_P', aliases[seed, 'F', family, beta]['condition'],
                          aliases[seed, 'P', family, beta]['condition'], 'F_minus_P', family, beta))
    rows = []
    for seed, name, left, right, interface, family, beta in contrasts:
        for role in ('utility', 'native', 'audit'):
            roles = ([(v, t) for v, ts in AUDIT_ROLES.items() for t in ts] if role == 'audit'
                     else [(UTILITY_VIEW[t], t) for t in (SOURCE_TASKS if role == 'native' else UTILITY_TASKS)])
            for view, target in roles:
                for budget, scope in (itertools.product(rules['audit_budgets'], rules['audit_scopes']) if role == 'audit' else [(None, role)]):
                    a = selected(index, seed, left, role, view, target, budget, scope)
                    b = selected(index, seed, right, role, view, target, budget, scope)
                    for raw_split, split in historical.SPLITS.items():
                        use_split = raw_split.replace('validation', 'source_validation') if role == 'native' else raw_split
                        av, bv = historical.loss(a, use_split), historical.loss(b, use_split)
                        value = difference(av, bv)
                        rows.append(dict(seed=seed, comparison=name, interface=interface, family=family, beta=beta,
                            left=left, right=right, role=role, view=view, target=target, audit_budget=budget, scope=scope,
                            split=use_split if role == 'native' and raw_split.startswith('validation') else split,
                            left_loss=av, right_loss=bv, left_minus_right=value,
                            signed_gain_left_minus_right=-value if role == 'audit' and value is not None else None,
                            present=a is not None and b is not None))
    return rows


def vector_tradeoffs(cube, roundoff=1e-12):
    """One fixed pair per seed/scope/budget/weight; no delta-based dominance."""
    rows = []
    for row in cube:
        if row['attribute'] != 'SEX' or row['panel'] != 'full_authorized' or row['delta'] != 0:
            continue
        utility = row['utility_differences'];gains = row['audit_gain_differences']
        components = {'utility/'+k: v for k, v in utility.items()} | {'gain/'+k: v for k, v in gains.items()}
        missing = [k for k, v in components.items() if not finite(v)]
        worse = [k for k, v in components.items() if finite(v) and v > roundoff]
        better = [k for k, v in components.items() if finite(v) and v < -roundoff]
        rows.append({k: row[k] for k in ('seed', 'interface', 'J_beta', 'Iplus_beta', 'J_condition', 'Iplus_condition', 'weighting', 'scope', 'audit_budget')} | {
            'utility_differences': utility, 'audit_gain_differences': gains,
            'components_worse_for_J': worse, 'components_better_for_J': better, 'missing_components': missing,
            'J_dominates_Iplus_numeric': None if missing else not worse and bool(better),
            'Iplus_dominates_J_numeric': None if missing else not better and bool(worse),
            'roundoff_tolerance': roundoff, 'delta_used_for_dominance': False,
            'assessment_scope': 'complete numerical utility/recovery vector; no all-target or race-support certificate'})
    return rows


def pareto_rows(points, entries, rules):
    rows = []
    conditions = {e['condition']: e for e in entries if e['seed'] == 0}
    for weighting, scope, budget, domain, vector in itertools.product(rules['weights'], rules['audit_scopes'],
            rules['audit_budgets'], ('F', 'P', 'F_and_P'), rules['pareto']['vectors']):
        names = [c for c, e in conditions.items() if domain == 'F_and_P' or e['interface'] == domain]
        components = rules['pareto']['vectors'][vector]
        per_seed = {}
        for seed in rules['seeds']:
            vectors = []
            for condition in names:
                point = points[seed, condition, weighting, scope, budget]
                value = {'condition': condition,
                    **{'utility/'+k: v for k, v in point['utility'].items()},
                    **{'gain/'+k: v for k, v in point['gains'].items()}}
                vectors.append(value)
                per_seed[seed, condition] = value
            rows.extend(dict(seed=seed, aggregation='per_seed', domain=domain, vector=vector,
                weighting=weighting, scope=scope, audit_budget=budget, **r)
                for r in pareto_membership(vectors, components, rules['pareto']['roundoff_tolerance']))
        means = []
        for condition in names:
            value = {'condition': condition}
            for component in components:
                values = [per_seed[seed, condition].get(component) for seed in rules['seeds']]
                value[component] = statistics.mean(values) if all(finite(v) for v in values) else None
            means.append(value)
        rows.extend(dict(seed=None, aggregation='three_seed_mean', domain=domain, vector=vector,
            weighting=weighting, scope=scope, audit_budget=budget, **r)
            for r in pareto_membership(means, components, rules['pareto']['roundoff_tolerance']))
    return rows


def gzip_export(path):
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=9, mtime=0) as compressed:
        compressed.write(path.read_bytes())
    target = path.with_name(path.name+'.gz');target.write_bytes(stream.getvalue())
    return {'plain_sha256': sha(path), 'gzip_sha256': sha(target), 'plain_bytes': path.stat().st_size,
            'gzip_bytes': target.stat().st_size, 'published_file': target.name, 'plain_file_local_only': True}


def metric_rows(index, entries, rules):
    rows = []
    units = [(e['seed'], e['condition']) for e in entries]+[(s, 'E') for s in rules['seeds']]
    for seed, condition in units:
        for role in ('utility', 'native', 'audit'):
            if role == 'native' and condition == 'E':
                continue
            roles = ([(v, t) for v, ts in AUDIT_ROLES.items() for t in ts] if role == 'audit'
                     else [(UTILITY_VIEW[t], t) for t in (SOURCE_TASKS if role == 'native' else UTILITY_TASKS)])
            for view, target in roles:
                for budget, scope in (itertools.product(rules['audit_budgets'], SCOPES) if role == 'audit' else [(None, role)]):
                    row = selected(index, seed, condition, role, view, target, budget, scope)
                    prior = historical.control_row(index, seed, 'prior', target, budget) if role == 'audit' else None
                    for raw_split, split in historical.SPLITS.items():
                        use_split = raw_split.replace('validation', 'source_validation') if role == 'native' else raw_split
                        value, reference = historical.loss(row, use_split), historical.loss(prior, raw_split)
                        rows.append(dict(seed=seed, condition=condition, role=role, view=view, target=target,
                            audit_budget=budget, scope=scope,
                            split=use_split if role == 'native' and raw_split.startswith('validation') else split,
                            log_loss=value, prior_loss=reference, signed_prior_relative_gain=difference(reference, value),
                            selected_candidate=row['candidate_id'] if row else None, present=row is not None,
                            origin_metrics=row.get('origin_metrics') if row else None))
    return rows


def audit_changes(index, entries, rules):
    budgets, scopes, singletons = [], [], []
    for seed, condition in [(e['seed'], e['condition']) for e in entries]+[(s, 'E') for s in rules['seeds']]:
        for view, targets in AUDIT_ROLES.items():
            for target in targets:
                for scope in SCOPES:
                    a, b = [selected(index, seed, condition, 'audit', view, target, budget, scope) for budget in (120, 360)]
                    for raw_split, split in historical.SPLITS.items():
                        av, bv = historical.loss(a, raw_split), historical.loss(b, raw_split)
                        budgets.append(dict(seed=seed, condition=condition, view=view, target=target, scope=scope,
                            split=split, loss120=av, loss360=bv, loss360_minus120=difference(bv, av),
                            gain360_minus120=difference(av, bv), candidate120=a['candidate_id'] if a else None,
                            candidate360=b['candidate_id'] if b else None,
                            selected_epoch120=a['metadata'].get('selected_epoch') if a else None,
                            selected_epoch360=b['metadata'].get('selected_epoch') if b else None))
                for budget, (left, right) in itertools.product(rules['audit_budgets'], ((SCOPES[1], SCOPES[0]), (SCOPES[2], SCOPES[1]))):
                    a, b = [selected(index, seed, condition, 'audit', view, target, budget, scope) for scope in (left, right)]
                    for raw_split, split in historical.SPLITS.items():
                        scopes.append(dict(seed=seed, condition=condition, view=view, target=target, audit_budget=budget,
                            left_scope=left, right_scope=right, split=split,
                            signed_gain_left_minus_right=difference(historical.loss(b, raw_split), historical.loss(a, raw_split)),
                            left_candidate=a['candidate_id'] if a else None, right_candidate=b['candidate_id'] if b else None))
        for target, budget, scope, singleton in itertools.product(('SEX', 'RAC1P'), rules['audit_budgets'], SCOPES, ('A', 'B')):
            a = selected(index, seed, condition, 'audit', 'AB', target, budget, scope)
            b = selected(index, seed, condition, 'audit', singleton, target, budget, scope)
            for raw_split, split in historical.SPLITS.items():
                av, bv = historical.loss(a, raw_split), historical.loss(b, raw_split)
                if raw_split == 'validation' and finite(av) and finite(bv) and av > bv+1e-12:
                    raise ValueError('An inherited singleton was omitted from the coalition pool')
                singletons.append(dict(seed=seed, condition=condition, target=target, audit_budget=budget, scope=scope,
                    singleton=singleton, split=split, coalition_loss=av, singleton_loss=bv,
                    coalition_minus_singleton_loss=difference(av, bv), coalition_minus_singleton_gain=difference(bv, av),
                    development_monotonicity_required=False))
    return budgets, scopes, singletons


def contextual_rows(index, indices, entries, rules):
    references, contrasts = [], []
    for seed, release in itertools.product(rules['seeds'], historical.CONTEXT):
        for role, targets in (('transfer', UTILITY_TASKS), ('audit', ('SEX', 'RAC1P'))):
            for target, budget in itertools.product(targets, (rules['audit_budgets'] if role == 'audit' else (None,))):
                for old_scope in (('primary', 'pooled') if role == 'audit' else ('primary',)):
                    old = historical.old_selected(indices, seed, release, role, target, budget or 120, old_scope)
                    for raw_split, split in historical.SPLITS.items():
                        value = old.get(raw_split, {}).get('log_loss') if old else None
                        references.append(dict(seed=seed, historical_release=release, role=role, target=target,
                            report_audit_budget=budget, actual_audit_budget=old.get('evidence_audit_budget') if old else None,
                            original_scope=old_scope, split=split, log_loss=value,
                            original_candidate=old['candidate_id'] if old else None,
                            origin_metrics=old.get('origin_metrics') if old else None,
                            causal_replacement=False, respects_new_individual_policy='not established'))
                        if old_scope != 'primary':
                            continue
                        for e in (e for e in entries if e['seed'] == seed):
                            for scope in (SCOPES if role == 'audit' else ('utility',)):
                                view = 'AB' if role == 'audit' else UTILITY_VIEW[target]
                                learned_role = 'audit' if role == 'audit' else 'utility'
                                row = selected(index, seed, e['condition'], learned_role, view, target, budget, scope)
                                learned = historical.loss(row, raw_split)
                                contrasts.append(dict(seed=seed, left=e['condition'], right=release,
                                    comparison='contextual_learned_minus_historical', role=learned_role, view=view, target=target,
                                    audit_budget=budget, scope=scope, historical_scope='primary',
                                    historical_actual_audit_budget=old.get('evidence_audit_budget') if old else None,
                                    split=split, left_loss=learned, right_loss=value, left_minus_right=difference(learned, value),
                                    signed_gain_left_minus_right=difference(value, learned) if role == 'audit' else None,
                                    causal_replacement=False))
    return references, contrasts


def criteria_rows(index, history, indices, entries, aliases, rules):
    from scripts.summarize_acs_protection import audit_coverage as old_coverage
    old_meta = {b: history.selections(b) for b in rules['audit_budgets']}
    utility, policies, feature, support = [], [], [], []
    for seed in rules['seeds']:
        conditions = [e['condition'] for e in entries if e['seed'] == seed]+['E']
        for raw_split, split in historical.SPLITS.items():
            parent = {t: historical.old_loss(indices, seed, 'E_pca', 'transfer', t, raw_split, 120) for t in UTILITY_TASKS}
            banks = {r: historical.old_loss(indices, seed, r, 'transfer', 'same_residence', raw_split, 120)
                     for r in ('B_rich_bank', 'C_tree_bank')}
            utilities = {c: {t: historical.loss(selected(index, seed, c, 'utility', UTILITY_VIEW[t], t), raw_split)
                            for t in UTILITY_TASKS} for c in conditions}
            for condition in conditions:
                utility.append(dict(seed=seed, condition=condition, split=split,
                    **historical.utility_criteria(utilities[condition], parent, banks)))
            for budget in rules['audit_budgets']:
                priors = {t: historical.loss(historical.control_row(index, seed, 'prior', t, budget), raw_split) for t in ('SEX', 'RAC1P')}
                parent_attack = {t: historical.old_loss(indices, seed, 'E_pca', 'audit', t, raw_split, budget) for t in ('SEX', 'RAC1P')}
                parent_cov = {t: old_coverage(indices[budget], old_meta[budget][seed], seed, 'E_pca', t, raw_split, 'primary') for t in ('SEX', 'RAC1P')}
                for scope in SCOPES:
                    covered, attacked = {}, {}
                    for condition in conditions:
                        for view, targets in AUDIT_ROLES.items():
                            for target in targets:
                                row = selected(index, seed, condition, 'audit', view, target, budget, scope)
                                exposed = historical.control_row(index, seed, 'exposed', target, budget)
                                coverage = historical.audit_coverage(row, row.get('metadata', {}) if row else {}, raw_split, exposed, target)
                                covered[condition, view, target] = coverage
                                support.append(dict(seed=seed, condition=condition, view=view,
                                    audit_budget=budget, scope=scope, split=split, **coverage))
                            attacked[condition, view] = {t: historical.loss(selected(index, seed, condition, 'audit', view, t, budget, scope), raw_split) for t in ('SEX', 'RAC1P')}
                            cov = {t: historical.combined_coverage(covered[condition, view, t], parent_cov[t]) for t in ('SEX', 'RAC1P')}
                            policies.append(dict(seed=seed, condition=condition, view=view, audit_budget=budget, scope=scope,
                                split=split, fixed_parent='E_pca', parent_actual_audit_budget=120,
                                opposing_target_assessment='continuous scores only; no declared pass threshold',
                                **historical.parent_comparison(parent, utilities[condition], banks, parent_attack, attacked[condition, view], priors, cov)))
                    for family, beta, task in itertools.product(rules['families'], rules['beta_decimal_strings'], ('same_residence', 'commute_over20')):
                        left = aliases[seed, 'F', family, beta]['condition']
                        for comparator, access in itertools.product((aliases[seed, 'P', family, beta]['condition'], 'E'), (UTILITY_VIEW[task], 'AB')):
                            result = historical.feature_bank_comparison(utilities[left][task], utilities[comparator][task],
                                attacked[left, access], attacked[comparator, access], priors,
                                {t: historical.combined_coverage(covered[left, access, t], covered[comparator, access, t]) for t in ('SEX', 'RAC1P')})
                            feature.append(dict(seed=seed, family=family, beta=beta, left=left, comparator=comparator,
                                task=task, authorized_view=UTILITY_VIEW[task], sensitive_access=access, audit_budget=budget,
                                scope=scope, split=split, shared_zero_anchor_alias=beta == '0',
                                numeric_joint_inequalities=historical.joint_status([result['utility']['pass'],
                                    *[v['numeric_inequality'] for v in result['attributes'].values()]]),
                                separate_reserved_task_decision=True, **result))
    return utility, policies, feature, support


def training_evidence(out, evidence, entries, config):
    counts, gradients, curves = [], [], []
    for e in entries:
        path = ROOT/e['training_checkpoint']
        meta_path = path.parent.parent/'training.json' if e['reused'] else path.parent/'training.json'
        if not meta_path.exists():
            continue
        record = evidence.read(meta_path)
        meta = record['arms'][e['original_condition']] if e['reused'] else record
        source_passes, observer_passes = (0, 0) if e['reused'] else (80, 240)
        counts.append({**{k: e[k] for k in ('seed', 'condition', 'interface', 'family', 'beta', 'reused')},
            'training_record': str(meta_path.relative_to(ROOT)), 'training_record_sha256': sha(meta_path),
            'loaded_base_epochs': 60, 'loaded_observer_warmup_epochs': 20,
            'new_source_epochs': source_passes, 'new_observer_passes': observer_passes,
            'final_inherited_source_exposure_per_row': 140, 'final_observer_exposure_per_row': 260,
            'forward_parameters': 6355, 'counts': meta.get('counts'),
            'source_valid_label_exposures': meta.get('source_valid_label_exposures'),
            'observer_valid_label_exposures': meta.get('observer_valid_label_exposures'),
            'selected_epoch': meta.get('selected_epoch'), 'new_warmup_epochs': 0,
            'new_continuation_optimizer_steps': (meta['counts']['mapper_optimizer_steps']-
                meta['curve'][0]['counts']['mapper_optimizer_steps']) if not e['reused'] else 0,
            'new_observer_optimizer_steps': (meta['counts']['adversary_optimizer_steps']-
                meta['curve'][0]['counts']['adversary_optimizer_steps']) if not e['reused'] else 0})
        def visit(value, trail):
            if isinstance(value, dict):
                if 'groups' in value and 'coefficients' in value:
                    for group, data in value['groups'].items():
                        gradients.append(dict(seed=e['seed'], condition=e['condition'], diagnostic_path='/'.join(trail), group=group,
                            source_record=str(meta_path.relative_to(ROOT)), coefficients=value['coefficients'],
                            losses=value.get('losses'), state_unchanged=value.get('state_unchanged'), rng_unchanged=value.get('rng_unchanged'), **data))
                    return
                for key, item in value.items():
                    if key != 'curve':
                        visit(item, (*trail, key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    visit(item, (*trail, str(i)))
        visit(meta, ())
        curves.extend(dict(seed=e['seed'], condition=e['condition'], **point) for point in meta.get('curve', ()))
    save_csv(out/'TRAINING_COUNTS.csv', counts)
    save_csv(out/'GRADIENT_DIAGNOSTICS.csv', gradients)
    save_csv(out/'TRAINING_CURVES.csv', curves)
    return counts, gradients


def actual_fitting_counts(out):
    """Count actual unique fits from immutable completed metadata, not witnesses."""
    out = Path(out).resolve();manifest = read_json(out/'REUSE_MANIFEST.json')
    inputs = {str((out/'REUSE_MANIFEST.json').relative_to(ROOT)): sha(out/'REUSE_MANIFEST.json')}
    units = [];totals = dict(fresh_mlp_360_trajectories=0, catchup_360_trajectories=0, static_auditor_fits=0,
        utility_candidate_fits=0, new_source_epochs=0, new_observer_passes=0, new_mapper_optimizer_steps=0, new_observer_optimizer_steps=0)
    for entry in manifest['systems']:
        if entry['reused']:
            continue
        base = ROOT/entry['evidence_directory'];completion_path = base/'complete.json'
        if not completion_path.exists():
            continue
        completion = read_json(completion_path);inputs[str(completion_path.relative_to(ROOT))] = sha(completion_path)
        records = []
        for relative in ('audits/audit_selection.json', 'selection_before_test.json'):
            path = base/relative;key = str(path.relative_to(ROOT));digest = sha(path)
            if completion['files_sha256'][key] != digest:
                raise ValueError('Fitting-count input differs from immutable completion: '+key)
            inputs[key] = digest;records.append(read_json(path))
        audits, selections = records
        fits = {}
        for role, candidates in audits['candidates']['360'].items():
            for cid, meta in candidates.items():
                if meta.get('inherited_singleton') or meta.get('diagnostic_only'):
                    continue
                path = meta['base_candidate_directory']
                kind = ('catchup_360_trajectories' if meta['candidate_origin'] == 'own_catchup' else
                        'fresh_mlp_360_trajectories' if meta['family'] == 'mlp' else 'static_auditor_fits')
                if path in fits:
                    raise ValueError('An actual fit appears twice among non-inherited candidates')
                fits[path] = kind
        counted = {k: sum(v == k for v in fits.values()) for k in tuple(totals)[:3]}
        expected = audits['counts']
        assert counted == dict(fresh_mlp_360_trajectories=expected['new_fresh_mlp_trajectories'],
            catchup_360_trajectories=expected['own_catchup_trajectories'], static_auditor_fits=expected['new_static_candidates'])
        if set(selections['utility_metadata']) != {UTILITY_VIEW[t]+'/'+t for t in UTILITY_TASKS}:
            raise ValueError('Utility fitting-role count differs from the five authorized tasks')
        counted['utility_candidate_fits'] = sum(len(v['candidates']) for v in selections['utility_metadata'].values())
        path = (ROOT/entry['training_checkpoint']).parent/'training.json';inputs[str(path.relative_to(ROOT))] = sha(path)
        training = read_json(path);epochs = training['selected_epoch']
        counted.update(new_source_epochs=epochs, new_observer_passes=epochs*training['config']['adversary_updates_per_mapper_step'],
            new_mapper_optimizer_steps=training['counts']['mapper_optimizer_steps']-training['curve'][0]['counts']['mapper_optimizer_steps'],
            new_observer_optimizer_steps=training['counts']['adversary_optimizer_steps']-training['curve'][0]['counts']['adversary_optimizer_steps'])
        for key in totals:
            totals[key] += counted[key]
        units.append({k: entry[k] for k in ('seed', 'condition', 'interface', 'family', 'beta')} | counted)
    result = dict(complete=len(units) == 36, unique_paired_systems=54, reused_paired_systems=18,
        new_completed_paired_systems=len(units), new_fitting=totals, new_prefix_training_epochs=0,
        inherited_source_exposure_per_final_row=140, inherited_observer_exposure_per_final_row=260,
        inherited_prefix=dict(source_epochs=60, observer_warmup_epochs=20, newly_refitted=False),
        reused_direct_E_controls=3, reused_prior_and_exposed_control_sets=3, new_control_fits=0,
        deduplication='Count budget360 base_candidate_directory once; exclude inherited singleton witnesses and saved-only diagnostics. Budget120 is the same trajectory prefix.',
        per_unit=units, input_sha256=inputs, counting_source_sha256=sha(Path(__file__)))
    save_json(out/'FITTING_COUNTS.json', result)
    return result


def scalar_cell(values, digits=5):
    values = list(values)
    observed = [v for v in values if finite(v)]
    if not observed:
        return 'unavailable'
    value = f'{statistics.mean(observed):.{digits}f}'
    if len(observed) > 1:
        value += f' ± {statistics.stdev(observed):.{digits}f}'
    return value+(f' ({len(observed)}/3; partial)' if len(observed) != 3 else '')


def condition_label(entry):
    if entry['condition'] == 'E':
        return 'E fixed teacher (context)'
    return entry['interface']+' I (shared β=0)' if entry['beta'] == 0 else f"{entry['interface']} {entry['family']} β={beta_key(entry['beta'])}"


def primary_rows(pair_aggregate, **overrides):
    filters = dict(attribute='SEX', scope='expanded_catchup', audit_budget=360,
                   panel='residential_transfer', delta=.001, weighting='unweighted', interface='F')
    filters.update(overrides)
    return [r for r in pair_aggregate if all(r[k] == v for k, v in filters.items())]


def table_report(out, points, entries, rules, pairs, complete):
    conditions = [e for e in entries if e['seed'] == 0]+[dict(condition='E', interface='E', family='direct_teacher', beta=None)]
    lines = ['# Fixed coalition-strength comparison', '',
        ('The full 54-system matrix is complete: 18 historical systems and 36 new continuations.' if complete else
         'PARTIAL: unavailable new systems are retained as missing in every fixed comparison.'), '',
        'These are development-evaluation losses and signed prior-relative attack gains, in nats. '
        'Lower utility loss and lower attack gain are better. Each value is mean ± sample SD across the same three cohort-sharing seeds; '
        'the SD is descriptive. Unweighted validation selects each predictor once; PWGTP scores those same predictions. '
        'Display rounding does not enter any comparison; full saved values and the separate 1e−12 roundoff rule are used. '
        'The tables use the expanded catch-up-inclusive 360-epoch audit. Other scopes and both budgets remain in '
        '[the complete selected results](PER_SEED.csv) and [aggregates](AGGREGATE.csv).', '',
        'Source feasibility requires **each** of the three independent source heads to be within .01 nats of its original '
        'same-seed, same-weight PCA32 head. It is not a source average, native-head comparison or statistical noninferiority test. '
        'The primary comparison is F, SEX, residential-transfer utility, δ=.001, unweighted. '
        'Every fixed J/local coefficient pair is reported; no coefficient or release is selected for use.', '',
        'β changes only the additional sensitive-attribute penalty. The ordinary individual protection term remains at −.1. '
        'Iplus applies the extra local sum M_A+M_B; J applies the coalition mean M_AB. '
        'At β=0 they are the same ordinary I system. F releases two feature vectors; P releases the matched A two-source and B one-source probabilities. '
        'The forward architecture, fitting data, source-loss weights and total fitting capacity are matched.', '']
    for weighting in rules['weights']:
        lines.extend(['## '+('PWGTP' if weighting == 'person_weighted' else 'Unweighted'), ''])
        body = []
        for entry in conditions:
            rows = [points[s, entry['condition'], weighting, 'expanded_catchup', 360] for s in rules['seeds']]
            body.append([condition_label(entry), str(sum(r['source_feasibility']['pass'] is True for r in rows))+'/3',
                *[scalar_cell(r['utility'][t] for r in rows) for t in UTILITY_TASKS],
                *[scalar_cell(r['gains']['AB/'+t] for r in rows) for t in ('SEX', 'RAC1P')]])
        lines.extend([historical.markdown_table(['System', 'Source pass', 'Income', 'Employment', 'Coverage',
            'Residence', 'Commute', 'AB SEX gain', 'AB RAC1P gain'], body), ''])
    lines.extend(['## Fixed pairs in the primary utility panel', '',
        'The direct E teacher is a reused context control with a different fitting history. Its reserved-task utility '
        'and source-floor failures are both retained; it is not an additional coefficient-grid system.', '',
        'Cells below give the number of the **same fixed pair’s three seeds** qualifying under close matching / directional comparison. '
        'Both systems must pass all source floors and J must have strictly lower AB SEX gain. '
        'Close matching requires every absolute utility difference ≤.001; directional comparison permits utility improvements '
        'of any size while limiting every deterioration to .001. This distinction prevents calling distant utility points matched.', ''])
    for interface, weighting in itertools.product(rules['interfaces'], rules['weights']):
        records = {(r['J_beta'], r['Iplus_beta']): r for r in primary_rows(pairs, interface=interface, weighting=weighting)}
        matrix = []
        for jb in rules['beta_decimal_strings']:
            cells = []
            for lb in rules['beta_decimal_strings']:
                r = records[jb, lb]
                cells.append(f"{r['close_qualifying_seed_count']}/3 / {r['directional_qualifying_seed_count']}/3"+
                    ('; partial' if r['close_assessable_seed_count'] != 3 else ''))
            matrix.append([jb, *cells])
        lines.extend([f"**{interface}; {'PWGTP' if weighting == 'person_weighted' else 'unweighted'}**", '',
            historical.markdown_table(['J β / Iplus β', *rules['beta_decimal_strings']], matrix), ''])
    lines.extend(['The zero entries in the family grid name the same physical I anchor. Their self-comparison cannot show a strict improvement. '
        'No seed-specific choice, interpolation, or average restricted to qualifying seeds is used. '
        'Race comparisons are aggregate numerical descriptions: census code 4 is absent from independent attacker fitting and '
        'attacker validation. All nine categories and pool-specific observer exposure remain reported, and full race assessment is unassessable.', '',
        '[All panels, deltas and audit-scope comparisons](MATCHING_ANALYSIS.md) · '
        '[Native source heads](NATIVE_SOURCE.md) · [Audit budgets and exposure](AUDIT_FINDINGS.md) · '
        '[Contextual references](CONTEXTUAL_REFERENCES.md) · [Frozen numerical rules](comparison_rules.json)', ''])
    (out/'TABLE.md').write_text('\n'.join(lines))


def matching_report(out, cube, aggregate, rules):
    lines = ['# Complete fixed-pair comparison', '',
        'The full cube has 57,600 per-seed rows: every 5×5 J/Iplus pair, both interfaces, all three seeds, '
        'both weightings, four panels, four deltas, three audit scopes, two budgets and each attribute separately. '
        '[UTILITY_MATCHES.csv.gz](UTILITY_MATCHES.csv.gz) retains all five utility differences, all eleven forbidden-role gain differences, '
        'the two independent source-floor checks, support flags and exact exclusion reasons. '
        '[UTILITY_MATCHES_AGGREGATE.csv.gz](UTILITY_MATCHES_AGGREGATE.csv.gz) holds every fixed-pair mean, SD and 0–3 qualifying count; '
        'all seed values are included even when that seed fails feasibility.', '',
        'The following compact view uses expanded catch-up-inclusive 360-epoch SEX recovery. Each cell reports '
        'the number of **fixed coefficient pairs** eligible or qualifying in 3/3, 2/3 or 1/3 seeds. '
        'Eligibility requires both source floors and the utility comparison; qualification additionally requires lower SEX gain. Counts describe this finite grid, '
        'not independent experiments, probabilities or a selected method. No-match findings distinguish close matching from directional comparisons.', '']
    for interface, weighting in itertools.product(rules['interfaces'], rules['weights']):
        body = []
        for panel, delta in itertools.product(rules['panels'], rules['delta_values']):
            values = primary_rows(aggregate, interface=interface, weighting=weighting, panel=panel, delta=delta)
            body.append([panel, delta, *[' / '.join(str(sum(r[kind+suffix] == n for r in values)) for n in (3, 2, 1))
                for kind, suffix in itertools.product(('close', 'directional'), ('_source_and_utility_eligible_seed_count', '_qualifying_seed_count'))]])
        lines.extend([f"## {interface}; {'PWGTP' if weighting == 'person_weighted' else 'unweighted'}", '',
            historical.markdown_table(['Utility panel', 'δ', 'Close eligible 3/2/1', 'Close lower-SEX 3/2/1',
                                      'Directional eligible 3/2/1', 'Directional lower-SEX 3/2/1'], body), ''])
    lines.extend(['## Every local comparator for each primary F J point', '',
        'All 25 comparisons are shown, including excluded points. Utility differences and gain differences are J minus Iplus. '
        'The table uses unweighted expanded catch-up-inclusive 360-epoch SEX results and the primary residential-transfer panel. '
        'Each exclusion cell lists the three seed-specific reasons; empty means that seed qualifies. '
        'The full cube supplies the PWGTP, independent-scope, 120-epoch and RAC1P counterparts without selecting a favorable scope.', ''])
    chosen = {(r['J_beta'], r['Iplus_beta']): r for r in primary_rows(aggregate)}
    source = {(r['J_beta'], r['Iplus_beta'], r['seed']): r for r in cube if r['interface'] == 'F' and r['weighting'] == 'unweighted'
        and r['attribute'] == 'SEX' and r['scope'] == 'expanded_catchup' and r['audit_budget'] == 360
        and r['panel'] == 'residential_transfer' and r['delta'] == .001}
    body = []
    for jb, lb in itertools.product(rules['beta_decimal_strings'], repeat=2):
        r = chosen[jb, lb]
        body.append([jb, lb, f"{r['close_qualifying_seed_count']}/3", f"{r['directional_qualifying_seed_count']}/3",
            scalar_cell(r['gain_difference_per_seed'].values()),
            scalar_cell(r['utility_differences']['same_residence']['per_seed'].values()),
            '; '.join(f"s{s}: "+(', '.join(source[jb, lb, s]['exclusion_reasons']['directional']) or 'qualifies') for s in rules['seeds'])])
    lines.extend([historical.markdown_table(['J β', 'Local β', 'Close', 'Directional', 'Δ SEX gain', 'Δ residence loss', 'Directional exclusions'], body), '',
        'Pareto membership in [NONDOMINATED_POINTS.csv.gz](NONDOMINATED_POINTS.csv.gz) uses ordinary componentwise dominance with 1e−12 roundoff, '
        'not the utility-matching δ. Every vector is explicitly named in the frozen rules. '
        'F, P and their union are separate domains; individual-seed and three-seed-mean frontiers remain separate. '
        'A projected two-axis plot cannot establish dominance in a five-task or all-forbidden-role vector. '
        'Missing components prevent a full-vector assessment, and numerical race nondominance does not repair category support. '
        '[VECTOR_TRADEOFF.csv](VECTOR_TRADEOFF.csv) lists every component that improves or worsens for each fixed J/local pair, '
        'including all individual forbidden targets. A SEX benefit alone is not an overall tradeoff improvement.', ''])
    (out/'MATCHING_ANALYSIS.md').write_text('\n'.join(lines))


def native_report(out, metrics, entries, rules):
    lines = ['# Fixed native source heads', '',
        'These fixed source heads are reported separately from the independent validation-selected utility heads. '
        'They do not substitute for source feasibility and no better-of-native-and-probe selection is used. '
        'Losses are development-evaluation mean ± sample SD across three cohort-sharing seeds.', '']
    index = {(r['seed'], r['condition'], r['target'], r['split']): r['log_loss'] for r in metrics if r['role'] == 'native'}
    for split in ('development_evaluation', 'development_evaluation_person_weighted'):
        body = [[condition_label(e), *[scalar_cell(index.get((s, e['condition'], t, split)) for s in rules['seeds'])
                                     for t in SOURCE_TASKS]] for e in entries if e['seed'] == 0]
        lines.extend(['## '+('PWGTP' if split.endswith('weighted') else 'Unweighted'), '',
            historical.markdown_table(['System', 'Income', 'Employment', 'Coverage'], body), ''])
    lines.extend(['All source-validation diagnostics and per-seed predictions scores are in [NATIVE_SOURCE.csv](NATIVE_SOURCE.csv). '
        'The source objective retains weights .25/.25/.5 and the recipient routing A: income/employment, B: coverage.', ''])
    (out/'NATIVE_SOURCE.md').write_text('\n'.join(lines))


def audit_report(out, evidence, entries, budget_rows, scopes, cube, rules):
    new_names = {(e['seed'], e['condition']) for e in entries if not e['reused']}
    actual = {}
    for row in evidence.rows:
        if (row['seed'], row['condition']) not in new_names or row['role'] != 'audit' or row['audit_budget'] != 360:
            continue
        meta = row['metadata'];path = meta.get('base_candidate_directory')
        if path and not meta.get('inherited_singleton'):
            actual.setdefault((path, meta.get('family')), row)
    selected360 = [r for r in evidence.rows if (r['seed'], r['condition']) in new_names and r['role'] == 'audit'
        and r['audit_budget'] == 360 and 'expanded_catchup' in r['selected_scopes']]
    catchup = [r for r in selected360 if 'catchup' in r['candidate_id']]
    zero = [r for r in catchup if r['metadata'].get('selected_epoch') == 0]
    body = []
    for interface, target, scope in itertools.product(rules['interfaces'], ('SEX', 'RAC1P'), SCOPES):
        values = [r for r in budget_rows if r['condition'].startswith(interface+'_') and r['view'] == 'AB'
            and r['target'] == target and r['scope'] == scope and r['split'] == 'development_evaluation']
        body.append([interface, target, scope, sum(r['candidate120'] != r['candidate360'] or r['selected_epoch120'] != r['selected_epoch360'] for r in values),
            sum(finite(r['gain360_minus120']) and abs(r['gain360_minus120']) > 1e-12 for r in values), len(values)])
    byidentity = {(r['seed'], r['interface'], r['J_beta'], r['Iplus_beta'], r['attribute'], r['weighting'], r['scope'], r['panel'], r['delta'], r['audit_budget']): r for r in cube}
    changes = []
    for identity, a in byidentity.items():
        if identity[-1] != 120:
            continue
        b = byidentity[(*identity[:-1], 360)]
        if any(a['qualifies_'+kind] != b['qualifies_'+kind] for kind in ('close', 'directional')):
            changes.append({k: a[k] for k in ('seed', *PAIR_FIELDS) if k != 'audit_budget'} | {
                'close120': a['qualifies_close'], 'close360': b['qualifies_close'],
                'directional120': a['qualifies_directional'], 'directional360': b['qualifies_directional'],
                'gain_difference120': a['gain_difference'], 'gain_difference360': b['gain_difference']})
    save_csv(out/'PAIR_BUDGET_CHANGES.csv', changes)
    lines = ['# Frozen-release audit findings', '',
        'All audit choices use attacker-validation log loss alone. The 120-epoch and 360-epoch choices are nested on each prescribed trajectory. '
        'Original terminal Adam/RNG checkpoints establish continuity; no validation-selected earlier checkpoint is called a continuous restart. '
        'Logistic and the two fixed tree configurations remain eligible at both budgets. '
        'The original PCA32 parent is a **120-epoch historical reference**, explicitly marked in contextual comparisons; '
        'it was not newly fitted at 360 epochs.', '',
        'Standard independent, expanded independent and expanded catch-up-inclusive pools are separate. '
        'The expanded F pool includes public native-head compositions with their inherited source-head exposure. '
        'AB includes every legal singleton sensitive candidate. Every composition stays within one frozen system: '
        'alternative coefficients, seeds and retrained versions are never concatenated or shared with an attacker.', '',
        f'Among newly fitted final-system expanded catch-up selections at 360 epochs, {len(catchup)}/{len(selected360)} winners '
        f'come from a saved-start trajectory ({len(catchup)}/{9*len(new_names)} observer-eligible roles); {len(zero)} select epoch 0. '
        'The all-role denominator includes the two reserved-target roles per system, which have no saved observer. The standalone saved-adversary diagnostic is excluded '
        'from selection, while its identical epoch-zero predictions are eligible inside catch-up under the inherited rule. '
        'A catch-up-inclusive advantage therefore does not necessarily result from additional optimizer steps. '
        'Representation-fitting observer exposure and direct-coordinate Adam reset remain explicit in the candidate metadata.', '',
        historical.markdown_table(['Interface', 'Target', 'Scope', 'Changed selected checkpoint', 'Changed gain', 'Endpoints'], body), '',
        f'There are {len(changes)} per-seed utility/attribute comparison rows whose close or directional qualification changes '
        'between audit budgets, across the entire predeclared cube. The same numerical event can appear in several panels or deltas; '
        'this is not a count of independent discoveries. [Every changed row](PAIR_BUDGET_CHANGES.csv) preserves its fixed coefficient pair. '
        '[All budget differences](AUDIT_BUDGET.csv) and [scope differences](AUDIT_SCOPE.csv) retain positive and negative development changes. '
        'Validation pool inclusion implies no required development-score monotonicity.', '',
        'Independent attacker fitting and attacker validation lack race census code 4. Some inherited observers have representation-pool '
        'exposure to that category; this does not establish held-out nine-category assessment. '
        'Prior and exposed-label controls, all nine class rows and their failures are retained. '
        'No opposing-source or reserved-task privacy threshold is invented. Full race and all-target protection remain unassessable.', '',
        '[PER_CANDIDATE.csv.gz](PER_CANDIDATE.csv.gz) · [PER_CLASS.csv.gz](PER_CLASS.csv.gz) · '
        '[AUDIT_CURVES.csv.gz](AUDIT_CURVES.csv.gz) · [CANDIDATE_LINEAGE.json.gz](CANDIDATE_LINEAGE.json.gz) · '
        '[FITTING.csv](FITTING.csv) · [TRAINING_COUNTS.csv](TRAINING_COUNTS.csv)', '',
        'New representation work is limited to 36 continuations ×80 source epochs and ×240 observer passes. '
        'The 60 source-warmup epochs and 20 observer-warmup epochs are inherited; reused .1 and zero systems receive no new fitting. '
        'The hash-bound count of actual completed new work is 1,188 fresh MLP trajectories, 324 saved-start trajectories, 1,782 logistic/tree fits '
        'and 360 utility candidates. Counts are independently deduplicated from all 36 completed audit/utility manifests. '
        'Linked singleton witnesses are reused computations, not additional fitted models. '
        'Actual completed counts and source/audit exposure are recorded in [FITTING_COUNTS.json](FITTING_COUNTS.json) and '
        '[TRAINING_COUNTS.csv](TRAINING_COUNTS.csv). Longer fitting remains a finite predictive audit, not a privacy guarantee.', '']
    (out/'AUDIT_FINDINGS.md').write_text('\n'.join(lines))
    return {'new_pooled_selections': len(selected360), 'new_catchup_winners': len(catchup),
            'new_epoch0_winners': len(zero), 'pair_budget_changes': len(changes)}


def plots(out, points, entries, aliases, rules, pairs, metrics, budgets, scopes, gradients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    import numpy as np
    folder = out/'figures';folder.mkdir(exist_ok=True)
    colors = {'J': '#2471a3', 'Iplus': '#b05a24'}
    beta = list(map(float, rules['beta_decimal_strings']))
    def point(seed, interface, family, b, weight, scope='expanded_catchup', budget=360):
        name = aliases[seed, interface, family, beta_key(b)]['condition']
        return points[seed, name, weight, scope, budget]
    def curve(ax, x, values, label, color, linestyle='-'):
        array = np.array([[float(v) if finite(v) else np.nan for v in row] for row in values])
        for i in range(3):
            ax.plot(x, array[:, i], '.', color=color, alpha=.35, markersize=4)
        means = [statistics.mean(row) if all(finite(v) for v in row) else np.nan for row in values]
        sds = [statistics.stdev(row) if all(finite(v) for v in row) else np.nan for row in values]
        ax.errorbar(x, means, yerr=sds, marker='o', markersize=3, linewidth=1.2, capsize=2,
                    label=label, color=color, linestyle=linestyle)
        ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(5));ax.tick_params(labelsize=8)
    def save(fig, name, caption=None):
        if (name.startswith(('utility_', 'feature_minus_prediction_', 'equal_strength_')) or
                name in ('audit_budget', 'audit_scopes', 'native_minus_probe', 'gradient_norms')):
            caption = (caption or '')+'\nPoints are evaluated coefficients; straight segments only guide the eye. Error bars are descriptive three-seed SD, not uncertainty intervals.'
        if caption:
            fig.text(.5, .005, caption, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045 if caption and '\n' in caption else .025 if caption else 0, 1, .965))
        fig.savefig(folder/(name+'.png'), dpi=180);plt.close(fig)
    for target in ('SEX', 'RAC1P'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for i, interface in enumerate(rules['interfaces']):
            for j, weight in enumerate(rules['weights']):
                ax = axes[i, j]
                for family in rules['families']:
                    xs, ys = [], []
                    for b in beta:
                        if b == 0 and family != 'J':
                            continue
                        records = [point(s, interface, family, b, weight) for s in rules['seeds']]
                        x = [r['utility']['same_residence'] for r in records];y = [r['gains']['AB/'+target] for r in records]
                        color = '#555555' if b == 0 else colors[family]
                        for s, r in enumerate(records):
                            if finite(x[s]) and finite(y[s]):
                                ax.scatter(x[s], y[s], s=26, color=color, alpha=.5,
                                    marker='o' if r['source_feasibility']['pass'] is True else 'x')
                        if all(finite(v) for v in (*x, *y)):
                            mx, my = statistics.mean(x), statistics.mean(y)
                            if b == 0:
                                ax.scatter([mx], [my], marker='D', color=color, s=26, label='I (shared β=0)')
                            else:
                                xs.append(mx);ys.append(my)
                            label_xy = (.02, .5) if b == 0 else (.125+.25*(beta.index(b)-1), .98 if family == 'J' else .02)
                            label = 'I 0' if b == 0 else f"{'J' if family == 'J' else 'I+'} {b:g}"
                            ax.annotate(f"{label} ({sum(r['source_feasibility']['pass'] is True for r in records)}/3)",
                                (mx, my), xytext=label_xy, textcoords='axes fraction', fontsize=7, color=color,
                                ha='left' if b == 0 else 'center', va='center',
                                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 1},
                                arrowprops={'arrowstyle': '-', 'lw': .4, 'color': color})
                    ax.scatter(xs, ys, marker='D', color=colors[family], s=26, label=family)
                ax.set(title=f"{interface}; {'PWGTP' if weight == 'person_weighted' else 'unweighted'}",
                    xlabel='Residence log loss (nats)', ylabel=f'AB {target} recovery gain (nats)')
                ax.margins(x=.12, y=.2);ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.grid(alpha=.2);ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(.99, .88))
        fig.suptitle(f'Development evaluation: {target}, expanded catch-up, 360 epochs')
        save(fig, 'tradeoff_'+target,
            'Small points: seeds (circle = all three source floors pass; × = fail). Diamonds: means, labeled β and source-pass n/3. Two-axis projections are not full Pareto tests.'+
            ('\nRAC1P full-schema assessment is unassessable: census code 4 is absent from independent attacker fitting and validation.' if target == 'RAC1P' else ''))
    for kind in ('close', 'directional'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, interface in enumerate(rules['interfaces']):
            for j, weight in enumerate(rules['weights']):
                ax = axes[i, j]
                values = {(r['J_beta'], r['Iplus_beta']): r for r in primary_rows(pairs, interface=interface, weighting=weight)}
                z = np.array([[values[jb, lb][kind+'_qualifying_seed_count'] for lb in rules['beta_decimal_strings']] for jb in rules['beta_decimal_strings']])
                ax.imshow(z, vmin=0, vmax=3, cmap='Blues')
                for (r, c), value in np.ndenumerate(z):
                    ax.text(c, r, str(value)+'/3', ha='center', va='center', color='white' if value >= 2 else 'black')
                ax.set(xticks=range(5), xticklabels=rules['beta_decimal_strings'], yticks=range(5), yticklabels=rules['beta_decimal_strings'],
                    xlabel='Iplus β', ylabel='J β', title=f"{interface}; {'PWGTP' if weight == 'person_weighted' else 'unweighted'}")
        fig.suptitle(f'{kind.title()} utility comparison: residential transfer, δ=.001; SEX expanded catch-up 360')
        save(fig, 'matching_'+kind, 'Every cell is one fixed coefficient pair across all three seeds. Both systems must pass each source floor. No seed-specific coefficient selection.')
    for interface in rules['interfaces']:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for i, weight in enumerate(rules['weights']):
            records = {(r['J_beta'], r['Iplus_beta']): r for r in primary_rows(pairs, interface=interface, weighting=weight)}
            for j, metric in enumerate(('close_source_and_utility_eligible_seed_count', 'directional_source_and_utility_eligible_seed_count', 'gain_difference_mean')):
                ax = axes[i, j]
                z = np.array([[records[jb, lb][metric] if records[jb, lb][metric] is not None else np.nan for lb in rules['beta_decimal_strings']] for jb in rules['beta_decimal_strings']])
                if j < 2:
                    ax.imshow(z, vmin=0, vmax=3, cmap='Blues')
                else:
                    limit = max(.001, float(np.nanmax(np.abs(z)))) if np.isfinite(z).any() else .001
                    ax.imshow(z, vmin=-limit, vmax=limit, cmap='coolwarm')
                for (r, c), value in np.ndenumerate(z):
                    text = (str(int(value))+'/3' if j < 2 else f'{value:+.4f}') if np.isfinite(value) else 'NA'
                    ax.text(c, r, text, ha='center', va='center', fontsize=8,
                            color='white' if np.isfinite(value) and (value >= 2 if j < 2 else abs(value) > .65*limit) else 'black')
                ax.set(xticks=range(5), xticklabels=rules['beta_decimal_strings'], yticks=range(5), yticklabels=rules['beta_decimal_strings'],
                    xlabel='Iplus β', ylabel='J β', title=('Close utility eligibility' if j == 0 else 'Directional utility eligibility' if j == 1 else 'J − Iplus SEX gain (nats)')+
                    ('; PWGTP' if weight == 'person_weighted' else '; U'))
        fig.suptitle(f'{interface}: separate utility eligibility and recovery differences; source + residence, δ=.001')
        save(fig, 'eligibility_and_SEX_'+interface, 'Eligibility includes both source-floor checks but no SEX improvement requirement. Gain means use all three seeds, including ineligible seeds; pooled 360 epochs.')
    for weight in rules['weights']:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for i, interface in enumerate(rules['interfaces']):
            for j, task in enumerate(UTILITY_TASKS):
                ax = axes[i, j]
                for family in rules['families']:
                    curve(ax, beta, [[point(s, interface, family, b, weight)['utility'][task] for s in rules['seeds']] for b in beta], family, colors[family])
                if task in SOURCE_TASKS:
                    parent = [rules['original_parent_metric_identity'][str(s)]['tasks'][task]['scores'][rules['weights'][weight]]+.01 for s in rules['seeds']]
                    ax.axhline(statistics.mean(parent), color='gray', linestyle='--', linewidth=.8, label='Mean PCA32 + .01')
                ax.set(title=interface+': '+task, xlabel='β', ylabel='Independent-head loss (nats)');ax.legend(fontsize=6)
        fig.suptitle('Development utility: '+('PWGTP' if weight == 'person_weighted' else 'unweighted'))
        save(fig, 'utility_'+weight, 'Points: all individual seeds. Error bars: descriptive three-seed SD. The mean parent line is a visual reference; source checks are per seed and per task.')
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for ax, task in zip(axes, UTILITY_TASKS):
            for family in rules['families']:
                values = [[difference(point(s, 'F', family, b, weight)['utility'][task], point(s, 'P', family, b, weight)['utility'][task]) for s in rules['seeds']] for b in beta]
                curve(ax, beta, values, family, colors[family])
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=task, xlabel='β', ylabel='F minus P loss (nats)');ax.legend(fontsize=8)
        fig.suptitle('Matched feature-versus-prediction utility: '+('PWGTP' if weight == 'person_weighted' else 'unweighted'))
        save(fig, 'feature_minus_prediction_'+weight, 'Each difference uses the same seed, family and coefficient. Residence and commute remain separate outcomes.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, target in enumerate(('SEX', 'RAC1P')):
        for j, weight in enumerate(rules['weights']):
            ax = axes[i, j]
            for interface, color in (('F', '#2471a3'), ('P', '#b05a24')):
                values = [[difference(point(s, interface, 'J', b, weight)['gains']['AB/'+target], point(s, interface, 'Iplus', b, weight)['gains']['AB/'+target]) for s in rules['seeds']] for b in beta]
                curve(ax, beta, values, interface, color)
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=target+'; '+('PWGTP' if weight == 'person_weighted' else 'unweighted'), xlabel='β', ylabel='J minus Iplus recovery gain (nats)');ax.legend()
    fig.suptitle('Equal-strength coalition-versus-local contrasts; expanded catch-up 360')
    save(fig, 'equal_strength_recovery', 'Negative values favor J in recovery alone. Utility/source feasibility is assessed separately; race category support remains incomplete.')
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for i, weight in enumerate(rules['weights']):
        for j, task in enumerate(UTILITY_TASKS):
            ax = axes[i, j]
            for interface, color in (('F', '#2471a3'), ('P', '#b05a24')):
                values = [[difference(point(s, interface, 'J', b, weight)['utility'][task], point(s, interface, 'Iplus', b, weight)['utility'][task]) for s in rules['seeds']] for b in beta]
                curve(ax, beta, values, interface, color)
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=task+('; PWGTP' if weight == 'person_weighted' else '; U'),
                xlabel='β', ylabel='J minus Iplus loss (nats)');ax.legend(fontsize=8)
    fig.suptitle('Equal-strength coalition-versus-local utility contrasts')
    save(fig, 'equal_strength_utility', 'All five tasks and all three seeds remain separate. Negative values favor J on that task; they do not establish a complete tradeoff improvement.')
    conditions = [e for e in entries if e['seed'] == 0]
    for target in ('SEX', 'RAC1P'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for i, interface in enumerate(rules['interfaces']):
            subset = [e for e in conditions if e['interface'] == interface]
            for j, weight in enumerate(rules['weights']):
                ax = axes[i, j]
                for view, color, offset in (('A', '#4393c3', -.22), ('B', '#d6604d', 0), ('AB', '#4d4d4d', .22)):
                    values = [[points[s, e['condition'], weight, 'expanded_catchup', 360]['gains'][view+'/'+target] for s in rules['seeds']] for e in subset]
                    means = [statistics.mean(v) if all(finite(x) for x in v) else np.nan for v in values]
                    ax.bar(np.arange(len(subset))+offset, means, width=.21, color=color, label=view)
                    for k, value in enumerate(values):
                        ax.scatter([k+offset]*3, [v if finite(v) else np.nan for v in value], s=9, color='black', alpha=.35)
                ax.set_xticks(range(len(subset)), [e['family']+' '+beta_key(e['beta']) for e in subset], rotation=45, ha='right')
                ax.set(title=interface+'; '+('PWGTP' if weight == 'person_weighted' else 'unweighted'), ylabel=target+' recovery gain (nats)');ax.grid(axis='y', alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Individual and coalition recovery; expanded catch-up 360')
        save(fig, 'individual_coalition_'+target, 'Every AB validation pool includes the legal singleton candidates. Development gains need not be monotone across recipients.'+
            ('\nRAC1P: independent fitting/validation lack census code 4; aggregate recovery does not establish full-schema protection.' if target == 'RAC1P' else ''))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, target in enumerate(('SEX', 'RAC1P')):
        for j, weight in enumerate(rules['weights']):
            ax = axes[i, j]
            for interface, color in (('F', '#2471a3'), ('P', '#b05a24')):
                for family, linestyle in (('J', '-'), ('Iplus', '--')):
                    values = [[difference(point(s, interface, family, b, weight, budget=360)['gains']['AB/'+target], point(s, interface, family, b, weight, budget=120)['gains']['AB/'+target]) for s in rules['seeds']] for b in beta]
                    curve(ax, beta, values, interface+' '+family, color, linestyle)
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=target+'; '+('PWGTP' if weight == 'person_weighted' else 'unweighted'), xlabel='β', ylabel='360 minus 120 recovery gain (nats)');ax.legend(fontsize=7)
    fig.suptitle('Nested audit-budget sensitivity; expanded catch-up')
    save(fig, 'audit_budget', 'Same prescribed trajectories and fixed candidate families. Development changes may have either sign; selection uses unweighted attacker validation only.')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, target in enumerate(('SEX', 'RAC1P')):
        for j, interface in enumerate(rules['interfaces']):
            ax = axes[i, j]
            for family, color in colors.items():
                for scope, label, linestyle in (('expanded_independent', 'expanded − standard', '--'), ('expanded_catchup', 'catch-up − expanded', '-')):
                    base = 'standard_independent' if scope == 'expanded_independent' else 'expanded_independent'
                    values = [[difference(point(s, interface, family, b, 'unweighted', scope)['gains']['AB/'+target], point(s, interface, family, b, 'unweighted', base)['gains']['AB/'+target]) for s in rules['seeds']] for b in beta]
                    curve(ax, beta, values, family+' '+label, color, linestyle)
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=interface+' '+target, xlabel='β', ylabel='Scope recovery-gain difference (nats)');ax.legend(fontsize=6)
    fig.suptitle('Audit-pool sensitivity; unweighted development evaluation, 360 epochs')
    save(fig, 'audit_scopes', 'PWGTP counterparts and every individual recipient are retained in AUDIT_SCOPE.csv. Public compositions retain their inherited head exposure.')
    for view, target in (('AB', 'SEX'), ('B', 'RAC1P')):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for i, interface in enumerate(rules['interfaces']):
            subset = [e for e in conditions if e['interface'] == interface]
            for j, weight in enumerate(rules['weights']):
                ax = axes[i, j]
                for scope, color, offset in zip(SCOPES, ('#66a61e', '#7570b3', '#d95f02'), (-.22, 0, .22)):
                    values = [[points[s, e['condition'], weight, scope, 360]['gains'][view+'/'+target] for s in rules['seeds']] for e in subset]
                    means = [statistics.mean(v) if all(finite(x) for x in v) else np.nan for v in values]
                    ax.bar(np.arange(len(subset))+offset, means, width=.21, color=color, label=scope)
                    for k, value in enumerate(values):
                        ax.scatter([k+offset]*3, [v if finite(v) else np.nan for v in value], s=9, color='black', alpha=.35)
                ax.set_xticks(range(len(subset)), [e['family']+' '+beta_key(e['beta']) for e in subset], rotation=45, ha='right')
                ax.set(title=interface+'; '+('PWGTP' if weight == 'person_weighted' else 'unweighted'), ylabel=f'{view} {target} recovery gain (nats)')
                ax.grid(axis='y', alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'Independent, expanded and catch-up selections: {view} {target}, 360 epochs')
        save(fig, 'selected_scopes_'+view+'_'+target, 'Small points: all three seeds; bars: means. Development gains need not increase with audit-pool size. Race category support remains incomplete.')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    native_index = {(r['seed'], r['condition'], r['target']): r['log_loss'] for r in metrics if r['role'] == 'native' and r['split'] == 'development_evaluation'}
    for i, interface in enumerate(rules['interfaces']):
        for j, task in enumerate(SOURCE_TASKS):
            ax = axes[i, j]
            for family in rules['families']:
                values = [[difference(native_index.get((s, aliases[s, interface, family, beta_key(b)]['condition'], task)), point(s, interface, family, b, 'unweighted')['utility'][task]) for s in rules['seeds']] for b in beta]
                curve(ax, beta, values, family, colors[family])
            ax.axhline(0, color='gray', linewidth=.8);ax.set(title=interface+' '+task, xlabel='β', ylabel='Native minus probe loss (nats)');ax.legend(fontsize=8)
    fig.suptitle('Native training heads and independent utility heads remain separate')
    save(fig, 'native_minus_probe', 'Development evaluation, unweighted. Source feasibility uses independent selected heads only; no better-of-head selection.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    diagnostic = {(r['seed'], r['condition'], r['diagnostic_path']): r for r in gradients if r['group'] == 'all_mappers'}
    for i, interface in enumerate(rules['interfaces']):
        for j, stage in enumerate(('fork_diagnostic', 'final_diagnostic')):
            ax = axes[i, j]
            for family, color in colors.items():
                for metric, label, linestyle in (('combined_protection_l2', 'combined', '-'),
                        ('coalition_applied_l2' if family == 'J' else 'extra_local_applied_l2', 'extra term', '--')):
                    values = [[diagnostic.get((s, aliases[s, interface, family, beta_key(b)]['condition'], stage), {}).get(metric)
                               for s in rules['seeds']] for b in beta]
                    curve(ax, beta, values, family+' '+label, color, linestyle)
            ax.set(title=interface+': '+('common frozen fork' if j == 0 else 'fixed final iterate'),
                   xlabel='β', ylabel='Applied mapper-gradient L2 norm');ax.legend(fontsize=8)
    fig.suptitle('Protection gradients at the predeclared fixed minibatch')
    save(fig, 'gradient_norms', 'Equal nominal extra coefficients do not imply equal gradient norms. No extra optimizer steps or diagnostic-based coefficient selection.')
    return [str(p.relative_to(out)) for p in sorted(folder.glob('*.png'))]


def summarize(out):
    started = time.perf_counter()
    executing_source_sha256 = sha(Path(__file__))
    out = Path(out).resolve()
    evidence, config, rules, entries, aliases, old_config = load_evidence(out)
    index = historical.build_index(evidence.rows)
    history, context = historical.load_context(old_config, evidence)
    points = points_from_index(index, entries, rules)
    cube = pair_cube(points, aliases, rules)
    pair_aggregate = fixed_seed_summary(cube, PAIR_FIELDS, tuple(rules['seeds']))
    metrics = metric_rows(index, entries, rules)
    aggregate = historical.aggregate_metrics(metrics)
    contrasts = fixed_contrasts(index, entries, aliases, rules)
    for row in contrasts:
        row['shared_zero_anchor_alias'] = row['beta'] == '0'
    contrast_fields = ('comparison', 'interface', 'family', 'beta', 'left', 'right', 'role', 'view', 'target', 'audit_budget', 'scope', 'split', 'shared_zero_anchor_alias')
    contrast_aggregate = historical.aggregate_rows(contrasts, contrast_fields)
    for row in contrast_aggregate:
        row['signed_gain_difference_mean'] = -row['mean'] if row['role'] == 'audit' and finite(row['mean']) else None
    pareto = pareto_rows(points, entries, rules)
    vector_comparisons = vector_tradeoffs(cube, rules['comparison_roundoff_tolerance'])
    utilities, policies, feature, support = criteria_rows(index, history, context, entries, aliases, rules)
    budgets, scopes, singletons = audit_changes(index, entries, rules)
    refs, contextual = contextual_rows(index, context, entries, rules)
    context_fields = ('comparison', 'left', 'right', 'role', 'view', 'target', 'audit_budget', 'scope', 'historical_scope', 'historical_actual_audit_budget', 'split', 'causal_replacement')
    contextual_aggregate = historical.aggregate_rows(contextual, context_fields)
    for row in contextual_aggregate:
        row['signed_gain_difference_mean'] = -row['mean'] if row['role'] == 'audit' and finite(row['mean']) else None
    historical.contextual_report(out, refs)
    counts = historical.export_candidates(out, evidence)
    training, gradients = training_evidence(out, evidence, entries, config)
    fitting_counts = actual_fitting_counts(out)
    evidence.inputs.add(out/'FITTING_COUNTS.json')
    native = [r for r in metrics if r['role'] == 'native']
    native_aggregate = [r for r in aggregate if r['role'] == 'native']
    exports = [('PER_SEED.csv', metrics), ('AGGREGATE.csv', aggregate), ('UTILITY_MATCHES.csv', cube),
        ('UTILITY_MATCHES_AGGREGATE.csv', pair_aggregate), ('PAIRED.csv', contrasts), ('PAIRED_AGGREGATE.csv', contrast_aggregate),
        ('NONDOMINATED_POINTS.csv', pareto), ('SUPPORT.csv', support), ('AUDIT_BUDGET.csv', budgets), ('AUDIT_SCOPE.csv', scopes),
        ('VECTOR_TRADEOFF.csv', vector_comparisons),
        ('COALITION_MINUS_SINGLETON.csv', singletons), ('CONTEXTUAL.csv', refs), ('CONTEXTUAL_PAIRED.csv', contextual),
        ('CONTEXTUAL_PAIRED_AGGREGATE.csv', contextual_aggregate), ('NATIVE_SOURCE.csv', native), ('NATIVE_AGGREGATE.csv', native_aggregate)]
    for name, rows in exports:
        save_csv(out/name, rows)
    save_json(out/'criteria.json', {'margins': historical.MARGINS, 'fixed_parent': 'E_pca', 'parent_actual_audit_budget': 120,
        'per_seed_split_utility': utilities, 'per_seed_view_budget_scope_policy': policies,
        'opposing_task_threshold': None, 'all_target_certificate': False})
    save_json(out/'feature_capability.json', {'margins': historical.MARGINS, 'tasks_decided_separately': True, 'per_seed': feature})
    save_json(out/'SOURCE_FEASIBILITY.json', [{k: p[k] for k in ('seed', 'condition', 'interface', 'family', 'beta', 'weighting', 'source_feasibility')}
        for p in points.values() if p['scope'] == 'expanded_catchup' and p['audit_budget'] == 360])
    complete = len(evidence.completed_systems) == 54
    table_report(out, points, entries, rules, pair_aggregate, complete)
    matching_report(out, cube, pair_aggregate, rules)
    native_report(out, metrics, entries, rules)
    audit_summary = audit_report(out, evidence, entries, budgets, scopes, cube, rules)
    figures = plots(out, points, entries, aliases, rules, pair_aggregate, metrics, budgets, scopes, gradients)
    compressed = {name: gzip_export(out/name) for name in COMPRESSED_EXPORTS}
    for path in (out/name for name in ('TABLE.md', 'MATCHING_ANALYSIS.md', 'NATIVE_SOURCE.md', 'AUDIT_FINDINGS.md', 'CONTEXTUAL_REFERENCES.md')):
        content = path.read_text()
        for name in COMPRESSED_EXPORTS:
            content = content.replace(']('+name+')', ']('+name+'.gz)')
        path.write_text(content)
    save_json(out/'COMPACT_EXPORTS.json', {'format': 'gzip level9, mtime0, empty original filename; no tar/archive bundle',
        'decompress': "gzip.decompress(Path('UTILITY_MATCHES.csv.gz').read_bytes())", 'files': compressed})
    source_paths = [Path(__file__), ROOT/'scripts/acs_coalition_strength_comparisons.py', ROOT/'scripts/summarize_acs_coalition.py',
        ROOT/'scripts/summarize_acs_protection.py', ROOT/'scripts/summarize_acs_pca16_init.py',
        ROOT/'scripts/summarize_acs_restricted.py', ROOT/'scripts/summarize_acs_selective.py', ROOT/'scripts/summarize_acs_preservation.py']
    main_conditions = []
    for e, weight in itertools.product((e for e in entries if e['seed'] == 0), rules['weights']):
        rows = [points[s, e['condition'], weight, 'expanded_catchup', 360] for s in rules['seeds']]
        main_conditions.append(dict(condition=e['condition'], interface=e['interface'], family=e['family'], beta=e['beta'],
            weighting=weight, source_pass_per_seed={r['seed']: r['source_feasibility']['pass'] for r in rows},
            utility={t: historical.statistics(r['utility'][t] for r in rows) for t in UTILITY_TASKS},
            AB_gains={t: historical.statistics(r['gains']['AB/'+t] for r in rows) for t in ('SEX', 'RAC1P')}))
    if sha(Path(__file__)) != executing_source_sha256:
        raise RuntimeError('Reporter source changed during execution; rerender before publication')
    result = dict(evaluation_status=config['evaluation_status'], complete=complete,
        completed_systems=[{'seed': s, 'condition': c} for s, c in sorted(evidence.completed_systems)],
        physical_system_count=len(evidence.completed_systems), expected_physical_systems=54, reused_systems=18,
        new_completed_systems=len(evidence.completed_systems)-18, shared_zero_anchor=True,
        pair_cube_rows=len(cube), fixed_pair_aggregate_rows=len(pair_aggregate), pareto_rows=len(pareto),
        metric_counts=counts, audit_summary=audit_summary, training_rows=len(training), gradient_rows=len(gradients),
        aggregate_metrics_file='AGGREGATE.csv', paired_aggregate_metrics_file='PAIRED_AGGREGATE.csv.gz',
        main_conditions=main_conditions,
        primary_fixed_pairs=primary_rows(pair_aggregate), compact_exports=compressed, figures=figures,
        report_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(evidence.inputs)},
        reporting_runtime_seconds=time.perf_counter()-started)
    save_json(out/'summary.json', result)
    print({k: result[k] for k in ('complete', 'physical_system_count', 'new_completed_systems', 'pair_cube_rows', 'fixed_pair_aggregate_rows', 'reporting_runtime_seconds')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)
