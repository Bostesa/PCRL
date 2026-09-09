"""Read-only source-guard comparisons, using the existing score/report helpers.

No raw data, fitted estimators, representation training, or predictor selection is
performed here. Validation and development are distinct comparison identities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import summarize_acs_coalition as historical
from scripts import summarize_acs_coalition_strength as previous
from scripts.acs_coalition_strength_comparisons import (
    SOURCE_TASKS, UTILITY_TASKS, AUDIT_ROLES, finite, difference, complete_and,
    evaluate_pair, fixed_seed_summary, source_check, pareto_membership,
)

ROOT = Path(__file__).resolve().parents[1]
SCOPES, UTILITY_VIEW = historical.SCOPES, historical.UTILITY_VIEW
read_json, save_json, save_csv, sha = historical.read_json, historical.save_json, historical.save_csv, historical.sha
ARMS = ('G_J', 'G_L025', 'G_L20', 'T', 'U_J', 'U_L025', 'U_L20', 'I')
NEW_ARMS = ARMS[:4]
GUARDED_ARMS = NEW_ARMS[:3]
PAIR_FIELDS = ('interface', 'J_arm', 'local_arm', 'J_condition', 'local_condition',
               'attribute', 'evaluation_split', 'weighting', 'scope', 'audit_budget', 'panel', 'delta')


def registry(manifest, rules):
    entries = manifest['systems']
    aliases = {}
    if len(entries) != 48 or len({(e['seed'], e['condition']) for e in entries}) != 48:
        raise ValueError('Expected 48 distinct interface systems, including 24 new systems')
    for entry in entries:
        key = entry['seed'], entry['interface'], entry['arm']
        if key in aliases:
            raise ValueError('Repeated fixed method identity')
        aliases[key] = entry
        if entry['reused'] != (entry['arm'] not in NEW_ARMS):
            raise ValueError('A new method was substituted with historical evidence')
    if set(aliases) != set(itertools.product(rules['seeds'], rules['interfaces'], ARMS)):
        raise ValueError('Missing or undeclared seed/interface/method identity')
    return entries, aliases


def score_key(rules, evaluation_split, weighting):
    return rules['evaluation_splits'][evaluation_split][weighting]


def parent_scores(rules, seed, evaluation_split, weighting):
    key = score_key(rules, evaluation_split, weighting)
    return {task: rules['original_parent_metric_identity'][str(seed)]['tasks'][task]['scores'][key]
            for task in SOURCE_TASKS}


def points_from_index(index, entries, rules):
    points = {}
    for entry, split, weight in itertools.product(entries, rules['evaluation_splits'], rules['weights']):
        seed, condition = entry['seed'], entry['condition']
        raw_split = score_key(rules, split, weight)
        utility = {task: historical.loss(historical.selected(index, seed, condition, 'utility',
            UTILITY_VIEW[task], task), raw_split) for task in UTILITY_TASKS}
        parent = parent_scores(rules, seed, split, weight)
        feasibility = source_check({'utility': utility}, parent, roundoff=rules['comparison_roundoff_tolerance'])
        for scope, budget in itertools.product(rules['audit_scopes'], rules['audit_budgets']):
            gains, losses, candidate_ids, coverage = {}, {}, {}, {}
            for view, targets in AUDIT_ROLES.items():
                for target in targets:
                    key = view+'/'+target
                    row = historical.selected(index, seed, condition, 'audit', view, target, budget, scope)
                    prior = historical.control_row(index, seed, 'prior', target, budget)
                    exposed = historical.control_row(index, seed, 'exposed', target, budget)
                    losses[key] = historical.loss(row, raw_split)
                    gains[key] = difference(historical.loss(prior, raw_split), losses[key])
                    candidate_ids[key] = row['candidate_id'] if row else None
                    coverage[key] = historical.audit_coverage(row, row.get('metadata', {}) if row else {}, raw_split, exposed, target)
            key = seed, condition, split, weight, scope, budget
            points[key] = {**entry, 'evaluation_split': split, 'raw_score_split': raw_split, 'weighting': weight,
                'scope': scope, 'audit_budget': budget, 'utility': utility, 'gains': gains,
                'attack_losses': losses, 'candidate_ids': candidate_ids, 'coverage': coverage,
                'source_feasibility': feasibility, 'parent': parent}
    return points


def pair_cube(points, aliases, rules):
    rows = []
    for seed, interface, pair, split, weight, scope, budget, panel, delta, attribute in itertools.product(
            rules['seeds'], rules['interfaces'], rules['fixed_method_pairs'], rules['evaluation_splits'],
            rules['weights'], rules['audit_scopes'], rules['audit_budgets'], rules['panels'], rules['delta_values'], ('SEX', 'RAC1P')):
        left, right = [aliases[seed, interface, arm] for arm in pair]
        a, b = [points[seed, e['condition'], split, weight, scope, budget] for e in (left, right)]
        result = evaluate_pair(a, b, a['parent'], rules['panels'][panel], delta, attribute,
            rules['comparison_roundoff_tolerance'])
        # Keep the validated pure arithmetic helper, but label every exclusion by
        # its actual guarded method rather than a misleading historical alias.
        result['exclusion_reasons'] = {kind: [reason.replace('Iplus/', pair[1]+'/').replace('J/', 'G_J/')
            for reason in reasons] for kind, reasons in result['exclusion_reasons'].items()}
        result['missing_required'] = [reason.replace('Iplus/', pair[1]+'/').replace('J/', 'G_J/')
            for reason in result['missing_required']]
        for old, new in (('source_J', 'source_guarded_J'), ('source_Iplus', 'source_guarded_local'),
                         ('J_gain', 'guarded_J_gain'), ('Iplus_gain', 'guarded_local_gain')):
            result[new] = result.pop(old)
        rows.append(dict(seed=seed, interface=interface, J_arm=pair[0], local_arm=pair[1],
            J_condition=left['condition'], local_condition=right['condition'], attribute=attribute,
            evaluation_split=split, weighting=weight, scope=scope, audit_budget=budget, panel=panel, delta=delta,
            J_selected_candidate=a['candidate_ids']['AB/'+attribute], local_selected_candidate=b['candidate_ids']['AB/'+attribute],
            J_attribute_support_complete=a['coverage']['AB/'+attribute]['complete'],
            local_attribute_support_complete=b['coverage']['AB/'+attribute]['complete'],
            attribute_full_schema_assessment='covered' if a['coverage']['AB/'+attribute]['complete'] and b['coverage']['AB/'+attribute]['complete'] else 'unassessable',
            qualification_scope='aggregate numerical inequalities; no race or all-target support certificate', **result))
    if len(rows) != rules['all_pair_grid']['total_rows']:
        raise ValueError('Incomplete fixed method comparison cube')
    return rows


def aggregate_pairs(rows, rules):
    result = fixed_seed_summary(rows, PAIR_FIELDS, tuple(rules['seeds']))
    if len(result) != rules['all_pair_grid']['fixed_pair_aggregate_rows']:
        raise ValueError('Incomplete fixed-pair aggregate identities')
    return result


def comparison_endpoints(aliases, rules):
    rows = []
    for seed, interface in itertools.product(rules['seeds'], rules['interfaces']):
        def add(name, left, right):
            rows.append(dict(seed=seed, interface=interface, comparison=name, left_arm=left, right_arm=right,
                left=aliases[seed, interface, left]['condition'], right=aliases[seed, interface, right]['condition']))
        for guarded, original in rules['guarded_unguarded_pairs']:
            add('guarded_minus_unguarded', guarded, original)
            add('guarded_minus_T', guarded, 'T')
        for left, right in rules['fixed_method_pairs']:
            add('guarded_J_minus_guarded_local', left, right)
        add('T_minus_I', 'T', 'I')
    for seed, arm in itertools.product(rules['seeds'], NEW_ARMS):
        rows.append(dict(seed=seed, interface='F_minus_P', comparison='F_minus_P', left_arm=arm, right_arm=arm,
            left=aliases[seed, 'F', arm]['condition'], right=aliases[seed, 'P', arm]['condition']))
    return rows


def fixed_contrasts(index, aliases, rules):
    rows = []
    for endpoint in comparison_endpoints(aliases, rules):
        seed, left, right = endpoint['seed'], endpoint['left'], endpoint['right']
        for role in ('utility', 'native', 'audit'):
            roles = ([(v, t) for v, ts in AUDIT_ROLES.items() for t in ts] if role == 'audit' else
                     [(UTILITY_VIEW[t], t) for t in (SOURCE_TASKS if role == 'native' else UTILITY_TASKS)])
            for view, target in roles:
                for budget, scope in (itertools.product(rules['audit_budgets'], rules['audit_scopes']) if role == 'audit' else [(None, role)]):
                    a = historical.selected(index, seed, left, role, view, target, budget, scope)
                    b = historical.selected(index, seed, right, role, view, target, budget, scope)
                    for split, weight in itertools.product(rules['evaluation_splits'], rules['weights']):
                        key = score_key(rules, split, weight)
                        if role == 'native':
                            key = key.replace('validation', 'source_validation')
                        av, bv = historical.loss(a, key), historical.loss(b, key)
                        delta = difference(av, bv)
                        rows.append({**endpoint, 'role': role, 'view': view, 'target': target, 'audit_budget': budget,
                            'scope': scope, 'evaluation_split': split, 'weighting': weight, 'raw_score_split': key,
                            'left_loss': av, 'right_loss': bv, 'left_minus_right': delta,
                            'signed_gain_left_minus_right': -delta if role == 'audit' and finite(delta) else None,
                            'present': a is not None and b is not None})
    return rows


def feasibility_rows(points, rules):
    return [{k: point[k] for k in ('seed', 'condition', 'interface', 'arm', 'reused', 'evaluation_split', 'weighting', 'source_feasibility')}
        for point in points.values() if point['scope'] == rules['primary']['audit_scope'] and point['audit_budget'] == rules['primary']['audit_budget']]


def vector_tradeoffs(cube, roundoff=1e-12):
    rows = []
    for row in cube:
        if row['attribute'] != 'SEX' or row['panel'] != 'full_authorized' or row['delta'] != 0:
            continue
        components = {'utility/'+k: v for k, v in row['utility_differences'].items()} | {
            'gain/'+k: v for k, v in row['audit_gain_differences'].items()}
        missing = [key for key, value in components.items() if not finite(value)]
        worse = [key for key, value in components.items() if finite(value) and value > roundoff]
        better = [key for key, value in components.items() if finite(value) and value < -roundoff]
        rows.append({k: row[k] for k in ('seed', 'interface', 'J_arm', 'local_arm', 'J_condition', 'local_condition',
            'evaluation_split', 'weighting', 'scope', 'audit_budget')} | dict(
            utility_differences=row['utility_differences'], audit_gain_differences=row['audit_gain_differences'],
            components_worse_for_guarded_J=worse, components_better_for_guarded_J=better,
            missing_components=missing, guarded_J_dominates_local_numeric=None if missing else not worse and bool(better),
            local_dominates_guarded_J_numeric=None if missing else not better and bool(worse),
            comparison_roundoff_tolerance=roundoff, delta_used_for_dominance=False,
            assessment_scope='numerical complete vector only; incomplete race support remains unassessable'))
    return rows


def _bound_unit_files(evidence, base, *, new, out=None):
    """Read the original completion manifest; check only consumed compact files."""
    if (base/'complete.json').exists():
        complete = evidence.read(base/'complete.json')
        hashes = complete['files_sha256']
        if not complete.get('evaluation_complete', False):
            raise ValueError('A condition completion record lacks complete evaluation')
        if new:
            release = out/'RELEASE_MANIFEST.json'
            if not release.exists() or sha(release) != complete['global_release_manifest_sha256']:
                raise ValueError('New metrics do not match the global release freeze')
            evidence.inputs.add(release)
        return hashes
    if new:
        return None
    # Earlier coalition units have a shared seed-level completion manifest.
    complete = evidence.read(base.parent/'complete.json')
    if not complete.get('six_pairs_complete', False):
        raise ValueError('Incomplete historical seed reference')
    return {str((base.parent/name).relative_to(ROOT)): digest for name,digest in complete['compact_sha256'].items()}


def add_unit(evidence, entry, out):
    base = ROOT/entry['evidence_directory']
    hashes = _bound_unit_files(evidence, base, new=not entry['reused'], out=out)
    if hashes is None:
        return False
    original = entry.get('original_condition', entry['condition'])
    record = previous.checked_read(evidence, base/'metrics.json', hashes)
    selection = previous.checked_read(evidence, base/'selection_before_test.json', hashes)
    audits = previous.checked_read(evidence, base/'audits/audit_selection.json', hashes)
    for raw in record['raw_metrics']:
        if raw['seed'] != entry['seed'] or raw['condition'] != original:
            raise ValueError('Raw candidate differs from its immutable unit identity')
        role = raw['view']+'/'+raw['target']
        if raw['role'] == 'audit':
            metadata = audits['candidates'][str(raw['audit_budget'])][role][raw['candidate_id']]
            expected = [scope for scope,cid in audits['selections'][str(raw['audit_budget'])][role].items() if cid == raw['candidate_id']]
            if set(expected) != set(raw['selected_scopes']):
                raise ValueError('An audit selection changed after validation freeze')
        elif raw['role'] == 'utility':
            if raw['view'] != UTILITY_VIEW[raw['target']]:
                raise ValueError('Unauthorized utility recipient')
            metadata = selection['utility_metadata'][role]['candidates'][raw['candidate_id']]
            if bool(raw['selected_scopes']) != (selection['utility'][role] == raw['candidate_id']):
                raise ValueError('Utility selection changed after validation freeze')
        else:
            raise ValueError('Undeclared raw result role')
        item = {**raw, 'condition':entry['condition'], 'original_condition':original, 'arm':entry['arm'], 'reused':entry['reused']}
        evidence.add(item, metadata, base/'metrics.json')
    native_path = base/'native_source.json'
    if not native_path.exists():
        native_path = base.parent/'native_source.json'
    native = previous.checked_read(evidence, native_path, hashes)
    for raw in previous.native_rows(native,original,entry['condition']):
        evidence.add({**raw,'arm':entry['arm'],'reused':entry['reused']}, {'family':'fixed_native_head'}, native_path)
    evidence.audit_manifests[entry['seed'],entry['condition']] = audits
    return True


def add_controls(evidence, out, config, seeds):
    historical_controls = {}
    for seed in seeds:
        directory = out/f'seed_{seed}'
        complete = evidence.read(directory/'complete.json')
        hashes = {str((directory/name).relative_to(ROOT)):digest for name,digest in complete['compact_sha256'].items()}
        base = directory/'E'
        audits = previous.checked_read(evidence, base/'audits/audit_selection.json', hashes)
        selection = previous.checked_read(evidence, base/'selection_before_test.json', hashes)
        record = previous.checked_read(evidence, base/'metrics.json', hashes)
        for raw in record['raw_metrics']:
            role = raw['view']+'/'+raw['target']
            if raw['role']=='audit':
                metadata = audits['candidates'][str(raw['audit_budget'])][role][raw['candidate_id']]
                expected = [scope for scope,cid in audits['selections'][str(raw['audit_budget'])][role].items() if cid==raw['candidate_id']]
                if set(expected)!=set(raw['selected_scopes']):
                    raise ValueError('Direct teacher audit selection differs from its freeze')
            else:
                metadata = selection['utility_metadata'][role]['candidates'][raw['candidate_id']]
                if raw['view']!=UTILITY_VIEW[raw['target']] or bool(raw['selected_scopes'])!=(selection['utility'][role]==raw['candidate_id']):
                    raise ValueError('Direct teacher utility selection differs from its freeze')
            evidence.add(raw,metadata,base/'metrics.json')
        path = directory/'controls/metrics.json'
        controls = previous.checked_read(evidence,path,hashes)
        metadata = previous.checked_read(evidence,directory/'controls/selection_before_test.json',hashes)
        for raw in controls['raw_metrics']:
            if raw['condition']=='prior':
                for budget in (120,360):
                    evidence.add({**raw,'audit_budget':budget,'actual_fitting_budget':None}, metadata['prior_metadata'][raw['target']],path)
            else:
                target,budget=raw['target'],raw['audit_budget'];source=metadata['metadata'][target]
                if source.get('reused'):
                    if seed not in historical_controls:
                        historical_controls[seed]=evidence.read(ROOT/config['preservation_reference_results']/'extended'/f'seed_{seed}'/'selection_before_test.json')
                    cm=historical_controls[seed]['budgets'][str(budget)]['fitting_records'][f'audit/exposed/{target}']['candidates'][raw['candidate_id']]
                else:
                    cm=source[f'nested{budget}']['candidates'][raw['candidate_id']]
                evidence.add(raw,cm,path)


def load_evidence(out):
    evidence=historical.Evidence()
    config=evidence.read(out/'config.json');rules=evidence.read(out/'comparison_rules.json')
    manifest=evidence.read(out/'REUSE_MANIFEST.json');freeze=evidence.read(out/'protocol_freeze.json')
    rule_path=str((out/'comparison_rules.json').relative_to(ROOT))
    if sha(out/'comparison_rules.json') != freeze['scientific_and_protocol_sha256'][rule_path]:
        raise ValueError('Prospectively frozen comparison rules changed')
    entries,aliases=registry(manifest,rules)
    completed=set()
    for entry in entries:
        if add_unit(evidence,entry,out):completed.add((entry['seed'],entry['condition']))
    add_controls(evidence,ROOT/config['coalition_reference_results'],config,rules['seeds'])
    for seed,identity in rules['original_parent_metric_identity'].items():
        for label in ('metrics','selection'):
            path=ROOT/identity[label+'_path']
            if sha(path)!=identity[label+'_sha256']:
                raise ValueError('Original selected PCA32 parent evidence changed')
            evidence.inputs.add(path)
    for name in ('PROTOCOL.md','PREFIT_REVIEW.md','EXECUTED_MATRIX.json','FITTING_COUNTS.json'):
        if (out/name).exists():evidence.inputs.add(out/name)
    evidence.completed_systems=completed
    return evidence,config,rules,entries,aliases


def pareto_rows(points, entries, rules):
    rows=[];conditions={e['condition']:e for e in entries if e['seed']==0}
    for split,weight,scope,budget,domain,vector in itertools.product(rules['evaluation_splits'],rules['weights'],
            rules['audit_scopes'],rules['audit_budgets'],('F','P','F_and_P'),rules['pareto']['vectors']):
        names=[name for name,e in conditions.items() if domain=='F_and_P' or e['interface']==domain]
        components=rules['pareto']['vectors'][vector];by_seed={}
        for seed in rules['seeds']:
            vectors=[]
            for name in names:
                point=points[seed,name,split,weight,scope,budget]
                value={'condition':name,**{'utility/'+k:v for k,v in point['utility'].items()},**{'gain/'+k:v for k,v in point['gains'].items()}}
                vectors.append(value);by_seed[seed,name]=value
            rows.extend(dict(seed=seed,aggregation='per_seed',evaluation_split=split,weighting=weight,scope=scope,
                audit_budget=budget,domain=domain,vector=vector,**r) for r in pareto_membership(vectors,components,rules['comparison_roundoff_tolerance']))
        means=[]
        for name in names:
            value={'condition':name}
            for component in components:
                values=[by_seed[seed,name].get(component) for seed in rules['seeds']]
                value[component]=statistics.mean(values) if all(finite(v) for v in values) else None
            means.append(value)
        rows.extend(dict(seed=None,aggregation='three_seed_mean',evaluation_split=split,weighting=weight,scope=scope,
            audit_budget=budget,domain=domain,vector=vector,**r) for r in pareto_membership(means,components,rules['comparison_roundoff_tolerance']))
    return rows


def criteria_rows(index, history, context, entries, aliases, rules):
    """Keep original source, residence, attribute and F/P task margins."""
    from scripts.summarize_acs_protection import audit_coverage as parent_coverage
    old_meta={budget:history.selections(budget) for budget in rules['audit_budgets']}
    utilities_out,policies,feature,support=[],[],[],[]
    for seed in rules['seeds']:
        conditions=[e['condition'] for e in entries if e['seed']==seed]+['E']
        for raw_split,split in historical.SPLITS.items():
            parent={t:historical.old_loss(context,seed,'E_pca','transfer',t,raw_split,120) for t in UTILITY_TASKS}
            banks={name:historical.old_loss(context,seed,name,'transfer','same_residence',raw_split,120) for name in ('B_rich_bank','C_tree_bank')}
            utility={condition:{t:historical.loss(historical.selected(index,seed,condition,'utility',UTILITY_VIEW[t],t),raw_split) for t in UTILITY_TASKS} for condition in conditions}
            for condition in conditions:
                utilities_out.append(dict(seed=seed,condition=condition,split=split,**historical.utility_criteria(utility[condition],parent,banks)))
            for budget in rules['audit_budgets']:
                priors={t:historical.loss(historical.control_row(index,seed,'prior',t,budget),raw_split) for t in ('SEX','RAC1P')}
                parent_attack={t:historical.old_loss(context,seed,'E_pca','audit',t,raw_split,budget) for t in ('SEX','RAC1P')}
                parent_cov={t:parent_coverage(context[budget],old_meta[budget][seed],seed,'E_pca',t,raw_split,'primary') for t in ('SEX','RAC1P')}
                for scope in rules['audit_scopes']:
                    covered,attacked={},{}
                    for condition in conditions:
                        for view,targets in AUDIT_ROLES.items():
                            for target in targets:
                                row=historical.selected(index,seed,condition,'audit',view,target,budget,scope)
                                exposed=historical.control_row(index,seed,'exposed',target,budget)
                                cov=historical.audit_coverage(row,row.get('metadata',{}) if row else {},raw_split,exposed,target)
                                covered[condition,view,target]=cov
                                support.append(dict(seed=seed,condition=condition,view=view,audit_budget=budget,scope=scope,split=split,**cov))
                            attacked[condition,view]={t:historical.loss(historical.selected(index,seed,condition,'audit',view,t,budget,scope),raw_split) for t in ('SEX','RAC1P')}
                            cov={t:historical.combined_coverage(covered[condition,view,t],parent_cov[t]) for t in ('SEX','RAC1P')}
                            policies.append(dict(seed=seed,condition=condition,view=view,audit_budget=budget,scope=scope,split=split,
                                fixed_parent='E_pca',parent_actual_audit_budget=120,opposing_target_assessment='continuous scores only; no invented pass threshold',
                                **historical.parent_comparison(parent,utility[condition],banks,parent_attack,attacked[condition,view],priors,cov)))
                    for arm,task in itertools.product(ARMS,('same_residence','commute_over20')):
                        left=aliases[seed,'F',arm]['condition']
                        for right,access in itertools.product((aliases[seed,'P',arm]['condition'],'E'),(UTILITY_VIEW[task],'AB')):
                            result=historical.feature_bank_comparison(utility[left][task],utility[right][task],attacked[left,access],attacked[right,access],priors,
                                {t:historical.combined_coverage(covered[left,access,t],covered[right,access,t]) for t in ('SEX','RAC1P')})
                            feature.append(dict(seed=seed,arm=arm,left=left,comparator=right,task=task,authorized_view=UTILITY_VIEW[task],sensitive_access=access,
                                audit_budget=budget,scope=scope,split=split,numeric_joint_inequalities=historical.joint_status([result['utility']['pass'],
                                    *[v['numeric_inequality'] for v in result['attributes'].values()]]),separate_reserved_task_decision=True,**result))
    return utilities_out,policies,feature,support


def native_report(out,index,entries,rules):
    lines=['# Native source heads and independent probes','',
        'The native heads are fixed training diagnostics. Source feasibility uses only the independent validation-selected 2,048-label utility probes, never a better-of-native-and-probe choice. Every cell is mean ± sample SD across the same three cohort-sharing seeds. These SDs are descriptive.','',
        'Validation columns use different pools: native heads use the original source-validation examples, while independent probes use downstream-validation examples. They are separate diagnostics, not paired head losses on the same examples. Development columns use the same held development examples.','']
    for split,weight in itertools.product(rules['evaluation_splits'],rules['weights']):
        key=score_key(rules,split,weight);native_key=key.replace('validation','source_validation')
        rows=[]
        for entry in (e for e in entries if e['seed']==0):
            values=[]
            for task in SOURCE_TASKS:
                native=[historical.loss(historical.selected(index,s,entry['condition'],'native',UTILITY_VIEW[task],task),native_key) for s in rules['seeds']]
                probe=[historical.loss(historical.selected(index,s,entry['condition'],'utility',UTILITY_VIEW[task],task),key) for s in rules['seeds']]
                values.extend((cell(native),cell(probe)))
            rows.append([entry['condition'],*values])
        lines.extend([f"## {split}; {'PWGTP' if weight=='person_weighted' else 'unweighted'}",'',
            historical.markdown_table(['System','Income native','Income probe','Employment native','Employment probe','Coverage native','Coverage probe'],rows),''])
    lines.extend(['[All per-seed native scores](NATIVE_SOURCE.csv) · [All selected probe scores](PER_SEED.csv) · [Paired system contrasts for native and probe scores](PAIRED.csv.gz)',''])
    (out/'NATIVE_SOURCE.md').write_text('\n'.join(lines))


def per_seed_reports(out,index,points,entries,rules):
    def scalar(value):return f'{value:.6f}' if finite(value) else 'unassessable'
    def flag(value):return 'PASS' if value is True else 'FAIL' if value is False else 'unassessable'
    utility=['# Every seed: independent utility and original source ceilings','',
        'Every primary system and seed is listed explicitly. Each source cell is `selected probe loss / original PCA32 loss + .01 / PASS or FAIL`, in nats. The ceiling uses that same seed, scoring split and weighting. Residence/commute cells are raw losses with no source-style threshold invented. Missing values remain unassessable. These tables use independent probes selected once by unweighted downstream validation; no native loss substitutes for a probe.','',
        'Validation is the downstream head-selection pool. Development evaluation is the previously inspected held cohort. They are distinct assessments, and PWGTP scores the same selected predictions.','']
    native=['# Every seed: native source heads and independent source probes','',
        'Native source heads are the fixed training heads. Probe heads use the independent original 2,048-label recipe. There is no better-of-head selection. Native source-validation and independent downstream-validation use different household pools; only the development columns compare the same held examples. All losses are nats.','']
    recovery=['# Every seed: all eleven forbidden recovery roles','',
        'Scope: expanded catch-up, genuine nested 360-epoch audit budget, development evaluation. All values are signed fitting-prior-minus-selected-attacker log-loss gains in nats. Higher means greater measured predictive recovery. A negative gain can reflect poorer development generalization than the prior; it is not negative information. PWGTP uses the same unweighted-validation-selected predictions.','',
        'These are the exact fixed primary-scope rows, not per-seed best configurations. All 120/360 budgets, three scopes, validation scores and selected candidate identities remain in [PER_SEED.csv](PER_SEED.csv). Race code 4 is absent from independent fitting and validation; full race assessment remains unassessable.','']
    for split,weight in itertools.product(rules['evaluation_splits'],rules['weights']):
        rows=[];native_rows=[];raw=score_key(rules,split,weight);native_key=raw.replace('validation','source_validation')
        for e in entries:
            p=points[e['seed'],e['condition'],split,weight,'expanded_catchup',360]
            source=[]
            for task in SOURCE_TASKS:
                f=p['source_feasibility']['tasks'][task]
                source.append(scalar(f['loss'])+' / '+scalar(f['parent_loss']+.01 if finite(f['parent_loss']) else None)+' / '+flag(f['pass']))
            rows.append([e['seed'],e['condition'],*source,scalar(p['utility']['same_residence']),scalar(p['utility']['commute_over20']),flag(p['source_feasibility']['pass'])])
            nv=[]
            for task in SOURCE_TASKS:
                nv.extend([scalar(historical.loss(historical.selected(index,e['seed'],e['condition'],'native',UTILITY_VIEW[task],task),native_key)),scalar(p['utility'][task])])
            native_rows.append([e['seed'],e['condition'],*nv])
        title='## '+split+'; '+weight
        utility.extend([title,'',historical.markdown_table(['Seed','System','Income loss / ceiling / flag','Employment loss / ceiling / flag','Coverage loss / ceiling / flag','Residence','Commute','All three source floors'],rows),''])
        native.extend([title,'',historical.markdown_table(['Seed','System','Income native','Income probe','Employment native','Employment probe','Coverage native','Coverage probe'],native_rows),''])
    for weight,view in itertools.product(rules['weights'],('A','B','AB')):
        rows=[]
        for e in entries:
            p=points[e['seed'],e['condition'],'development_evaluation',weight,'expanded_catchup',360]
            rows.append([e['seed'],e['condition'],*[scalar(p['gains'][view+'/'+target]) for target in AUDIT_ROLES[view]]])
        recovery.extend(['## '+view+'; '+weight,'',historical.markdown_table(['Seed','System',*AUDIT_ROLES[view]],rows),''])
    for name,lines in [('PER_SEED_UTILITY.md',utility),('PER_SEED_NATIVE.md',native),('PER_SEED_RECOVERY.md',recovery)]:
        (out/name).write_text('\n'.join(lines))


def controls_report(out,index,rules):
    """Read already-selected static controls; do not fit or select new models."""
    targets=('SEX','RAC1P','income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
    rows=[]
    for seed,target,budget,split,weight,condition in itertools.product(rules['seeds'],targets,rules['audit_budgets'],rules['evaluation_splits'],rules['weights'],('prior','exposed')):
        row=historical.control_row(index,seed,condition,target,budget)
        rows.append(dict(seed=seed,condition=condition,target=target,report_audit_budget=budget,
            actual_fitting_budget=None if condition=='prior' else budget,evaluation_split=split,weighting=weight,
            log_loss=historical.loss(row,score_key(rules,split,weight)),selected_candidate=row['candidate_id'] if row else None,
            origin_metrics=row.get('origin_metrics') if row else None,reused=True))
    save_csv(out/'CONTROL_RESULTS.csv',rows)
    lines=['# Reused direct teacher and audit controls','',
        'These are matching historical controls, reused without new fitting. The direct E teacher retains its original genuine nested 120/360-epoch audits and original utility heads. Priors have no fitting-epoch budget; repeating their score beside both audit budgets is an alias. Exposed-label controls retain the selected checkpoint from their declared original budget. Original PCA32 and other named references keep their actual historical audit budgets in [CONTEXTUAL_REFERENCES.md](CONTEXTUAL_REFERENCES.md).','',
        'All cells are mean ± descriptive sample SD across the same three cohort-sharing seeds. PWGTP uses exactly the predictors selected by unweighted validation. These are development-evaluation inputs previously used in method development.','', '## Direct E utility and coalition recovery','']
    table=[]
    for weight in rules['weights']:
        key=score_key(rules,'development_evaluation',weight)
        utility=[cell(historical.loss(historical.selected(index,seed,'E','utility',UTILITY_VIEW[task],task),key) for seed in rules['seeds']) for task in UTILITY_TASKS]
        table.append([weight,*utility])
    lines.extend([historical.markdown_table(['Weighting',*UTILITY_TASKS],table),''])
    table=[]
    for weight,budget,scope in itertools.product(rules['weights'],rules['audit_budgets'],rules['audit_scopes']):
        key=score_key(rules,'development_evaluation',weight)
        gains=[]
        for target in ('SEX','RAC1P'):
            gains.append(cell(difference(historical.loss(historical.control_row(index,seed,'prior',target,budget),key),historical.loss(historical.selected(index,seed,'E','audit','AB',target,budget,scope),key)) for seed in rules['seeds']))
        table.append([weight,budget,scope,*gains])
    lines.extend([historical.markdown_table(['Weighting','Actual nested budget','Scope','AB SEX signed gain','AB race signed gain'],table),'',
        '## Prior and exposed-target control losses','',
        'Lower exposed-label loss shows that the declared candidate/data recipe can recover an explicitly supplied target. A failure on a class remains a limitation; it is not evidence that a learned release protects that class. Full race assessment remains unassessable because code 4 is absent from independent fitting and validation. Inherited representation observer exposure is a separate route.',''])
    for budget,weight in itertools.product(rules['audit_budgets'],rules['weights']):
        table=[]
        for target in targets:
            values=[]
            for split,condition in itertools.product(rules['evaluation_splits'],('prior','exposed')):
                values.append(cell(r['log_loss'] for r in rows if r['target']==target and r['report_audit_budget']==budget and r['weighting']==weight and r['evaluation_split']==split and r['condition']==condition))
            table.append([target,*values])
        lines.extend([f'### Budget {budget}; {weight}','',historical.markdown_table(['Target','Validation prior','Validation exposed','Development prior','Development exposed'],table),''])
    lines.extend(['[All control rows and original predictor identities](CONTROL_RESULTS.csv) · [All direct E selected scores](PER_SEED.csv) · [All candidate and class evidence](PER_CANDIDATE.csv.gz) · [Category support](PER_CLASS.csv.gz)',''])
    (out/'STATIC_CONTROLS.md').write_text('\n'.join(lines))


def cell(values):
    values=list(values)
    return f'{statistics.mean(values):.5f} ± {statistics.stdev(values):.5f}' if len(values)==3 and all(finite(v) for v in values) else 'unassessable'


def primary_rows(rows,rules,*,interface='F',weighting='unweighted',split='development_evaluation',panel='residential_transfer',delta=.001,scope='expanded_catchup',budget=360,attribute='SEX'):
    return [r for r in rows if r['interface']==interface and r['weighting']==weighting and r['evaluation_split']==split and r['panel']==panel
        and r['delta']==delta and r['scope']==scope and r['audit_budget']==budget and r['attribute']==attribute]


def table_report(out,points,entries,rules,pairs,complete):
    evaluated_new=sum(not e['reused'] and all(finite(v) for v in points[e['seed'],e['condition'],'development_evaluation','unweighted','expanded_catchup',360]['utility'].values()) for e in entries)
    lines=['# Source-guard comparison','',
        ('The complete 48-system matrix contains 24 new interface systems and 24 matched reused systems.' if complete else
         f'**Partial result: {evaluated_new}/24 new interface systems evaluated.** Unfinished systems remain explicitly missing; no historical result substitutes for a missing guarded result.'),'',
        'Losses and signed prior-relative recovery gains are in nats; lower is better for both. Each cell is mean ± sample SD across three seeds sharing a cohort. Unweighted validation selected every utility/audit predictor once; PWGTP scores those same predictions. Native losses are separate diagnostics. A negative gain means the selected attacker has worse loss than the fitting-prior predictor on that scoring pool; it is not negative information or a privacy guarantee. No release was selected using downstream outcomes.','',
        'Source feasibility requires EACH selected independent source-probe loss ≤ its original same-seed, same-weight, same-split PCA32 loss +.01. Validation and development are separate checks; source averages and mean-only checks cannot substitute. The primary method comparison is F / AB SEX / expanded catch-up / 360 epochs / unweighted / development / source plus residence / δ=.001.','',
        'The two local comparators are fixed across seeds: G-L025 and G-L20. T removes all protection; ordinary I retains its original individual term. T has three shared forward trajectories but six distinct F/P release and observer/audit systems. Guarding a local current Adam displacement does not guarantee native loss or probe utility.','']
    for weight in rules['weights']:
        rows=[]
        for entry in (e for e in entries if e['seed']==0):
            records=[points[s,entry['condition'],'development_evaluation',weight,'expanded_catchup',360] for s in rules['seeds']]
            validation=[points[s,entry['condition'],'validation',weight,'expanded_catchup',360] for s in rules['seeds']]
            passes=lambda rr: str(sum(r['source_feasibility']['pass'] is True for r in rr))+'/3'+('' if complete else '; '+str(sum(r['source_feasibility']['pass'] is not None for r in rr))+'/3 assessed')
            rows.append([entry['condition'],passes(validation),passes(records),
                *[cell(r['utility'][task] for r in records) for task in UTILITY_TASKS],
                *[cell(r['gains']['AB/'+target] for r in records) for target in ('SEX','RAC1P')]])
        lines.extend(['## '+('PWGTP' if weight=='person_weighted' else 'Unweighted'),'',
            historical.markdown_table(['System','Probe validation pass','Development pass','Income','Employment','Coverage','Residence','Commute','AB SEX gain','AB race gain'],rows),''])
    rows=[]
    for interface,weight,pair in itertools.product(rules['interfaces'],rules['weights'],rules['fixed_method_pairs']):
        r=next(r for r in primary_rows(pairs,rules,interface=interface,weighting=weight) if r['local_arm']==pair[1])
        rows.append([interface,weight,pair[1],f"{r['close_source_and_utility_eligible_seed_count']}/3",f"{r['close_qualifying_seed_count']}/3",
            f"{r['directional_source_and_utility_eligible_seed_count']}/3",f"{r['directional_qualifying_seed_count']}/3",cell(r['gain_difference_per_seed'].values())])
    lines.extend(['## Fixed guarded J/local pairs in the primary utility panel','',
        'Close means every absolute panel loss difference ≤.001. Directional means every signed J-minus-local loss difference ≤.001; larger improvements are permitted. Eligibility requires both source floors and the utility condition; qualification adds strictly lower SEX gain. Counts are for one fixed pair across the same three seeds, never a stitched or qualifying-only average.','',
        historical.markdown_table(['Interface','Weighting','Fixed local','Close eligible','Close lower SEX','Directional eligible','Directional lower SEX','Δ SEX gain'],rows),'',
        'Race code 4 remains absent from independent attacker fitting and validation; full race assessment is unassessable. All nine categories, inherited observer exposure, signed negative gains and exposed-control failures remain reported. Separate saved-only diagnostics are excluded from selection; epoch 0 is eligible inside catch-up. Original PCA32 parent audits remain their actual historical 120-epoch reference.','',
        '[All paired tests and exclusions](MATCHING_ANALYSIS.md) · [Native/probe diagnostics](NATIVE_SOURCE.md) · [Individual forbidden roles](INDIVIDUAL_TARGETS.md) · [Audits](AUDIT_FINDINGS.md) · [Frozen rules](comparison_rules.json) · [Reused direct E and controls](STATIC_CONTROLS.md) · [Every seed: utility and source ceilings](PER_SEED_UTILITY.md) · [Every seed: all eleven forbidden roles](PER_SEED_RECOVERY.md) · [Every seed: native and probe source losses](PER_SEED_NATIVE.md)',''])
    (out/'TABLE.md').write_text('\n'.join(lines))


def matching_report(out,cube,aggregate,rules):
    lines=['# Fixed guarded-method utility matching','',
        'The complete comparison contains 9,216 per-seed rows and 3,072 fixed-pair aggregates: both fixed guarded local comparators, all four utility panels, four δ values, both attributes, three audit scopes, both budgets, both weightings and validation/development separately. [UTILITY_MATCHES.csv.gz](UTILITY_MATCHES.csv.gz) retains every task-specific exclusion; [UTILITY_MATCHES_AGGREGATE.csv.gz](UTILITY_MATCHES_AGGREGATE.csv.gz) includes every seed and all-three means/SD without filtering by eligibility.','',
        'Tables below use development AB SEX and expanded catch-up at 360 epochs. Each entry remains a fixed method pair. Validation and other audit scopes/budgets remain separate in the complete exports. Downstream validation is the independent probe-selection pool and is diagnostic, not an independent confirmation. It is distinct from the native source-validation pool.','']
    for interface,weight in itertools.product(rules['interfaces'],rules['weights']):
        rows=[]
        for panel,delta,pair in itertools.product(rules['panels'],rules['delta_values'],rules['fixed_method_pairs']):
            r=next(r for r in primary_rows(aggregate,rules,interface=interface,weighting=weight,panel=panel,delta=delta) if r['local_arm']==pair[1])
            rows.append([panel,delta,pair[1],f"{r['close_source_and_utility_eligible_seed_count']}/3",f"{r['close_qualifying_seed_count']}/3",
                f"{r['directional_source_and_utility_eligible_seed_count']}/3",f"{r['directional_qualifying_seed_count']}/3"])
        lines.extend([f"## {interface}; {weight}",'',historical.markdown_table(['Panel','δ','Local','Close eligible','Close lower SEX','Directional eligible','Directional lower SEX'],rows),''])
    lines.extend(['## Primary fixed-pair exclusions, every seed',''])
    rows=[]
    for r in cube:
        if r['attribute']=='SEX' and r['scope']=='expanded_catchup' and r['audit_budget']==360 and r['panel']=='residential_transfer' and r['delta']==.001 and r['evaluation_split']=='development_evaluation':
            rows.append([r['interface'],r['weighting'],r['local_arm'],r['seed'],r['qualifies_close'],r['qualifies_directional'],
                '; '.join(r['exclusion_reasons']['close']) or 'qualifies','; '.join(r['exclusion_reasons']['directional']) or 'qualifies'])
    lines.extend([historical.markdown_table(['Interface','Weighting','Local','Seed','Close','Directional','Close exclusions','Directional exclusions'],rows),'',
        '[Full vector tradeoffs](VECTOR_TRADEOFF.csv) keep all five authorized utility losses and eleven forbidden recovery gains. [Pareto evidence](NONDOMINATED_POINTS.csv.gz) uses ordinary componentwise dominance with 1e−12 roundoff, never δ; per-seed and three-seed-mean results are separate and missing components remain unassessable. A numerical race vector is not a full-support certificate.',''])
    (out/'MATCHING_ANALYSIS.md').write_text('\n'.join(lines))


def individual_report(out,points,entries,rules):
    lines=['# Individual forbidden-target recovery','',
        'Each value is development signed prior-relative gain in nats, mean ± descriptive sample SD over three cohort-sharing seeds. Higher means greater measured recovery. The table uses expanded catch-up at 360 epochs; all scopes, both budgets and validation are in PER_SEED.csv. No pass threshold is invented for opposing source or reserved tasks.','']
    for weight,view in itertools.product(rules['weights'],('A','B')):
        rows=[]
        for entry in (e for e in entries if e['seed']==0):
            records=[points[s,entry['condition'],'development_evaluation',weight,'expanded_catchup',360] for s in rules['seeds']]
            rows.append([entry['condition'],*[cell(r['gains'][view+'/'+target] for r in records) for target in AUDIT_ROLES[view]]])
        lines.extend([f'## {view}; {weight}','',historical.markdown_table(['System',*AUDIT_ROLES[view]],rows),''])
    lines.extend(['[Every seed and all eleven primary-scope recovery roles](PER_SEED_RECOVERY.md) are shown in readable tables. Standalone saved observers are diagnostics; epoch 0 inside catch-up remains eligible. Inherited observer/head exposure and missing independent race code-4 support remain explicit in [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md).',''])
    (out/'INDIVIDUAL_TARGETS.md').write_text('\n'.join(lines))


def plots(out,points,entries,aliases,rules,pairs,metrics,guard_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={'G_J':'#b2182b','G_L025':'#2166ac','G_L20':'#4393c3','T':'#444444',
            'U_J':'#ef8a62','U_L025':'#67a9cf','U_L20':'#92c5de','I':'#999999'}
    short={'G_J':'G-J','G_L025':'G-L025','G_L20':'G-L20','T':'T','U_J':'U-J','U_L025':'U-L025','U_L20':'U-L20','I':'I'}
    def point(seed,interface,arm,weight,split='development_evaluation',scope='expanded_catchup',budget=360):
        return points[seed,aliases[seed,interface,arm]['condition'],split,weight,scope,budget]
    def save(fig,name,caption):
        fig.text(.5,.008,caption,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(folder/(name+'.png'),dpi=180);plt.close(fig)
    def values_bar(ax,values,labels,colors=None):
        means=[statistics.mean(v) if all(finite(x) for x in v) else np.nan for v in values]
        errors=[statistics.stdev(v) if all(finite(x) for x in v) else np.nan for v in values]
        for i,vs in enumerate(values):
            if finite(means[i]):
                ax.bar(i,means[i],yerr=errors[i],color=colors[i] if colors is not None else '#4393c3',width=.65,alpha=.8,capsize=2)
            ax.scatter([i-.12,i,i+.12],[v if finite(v) else np.nan for v in vs],color='black',s=13,zorder=3)
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.axhline(0,color='gray',lw=.8);ax.grid(axis='y',alpha=.2)
    for interface in rules['interfaces']:
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for i,split in enumerate(rules['evaluation_splits']):
            for j,weight in enumerate(rules['weights']):
                ax=axes[i,j]
                for t,(task,color) in enumerate(zip(SOURCE_TASKS,('#1b9e77','#d95f02','#7570b3'))):
                    xx=[];yy=[]
                    for k,arm in enumerate(ARMS):
                        vs=[point(s,interface,arm,weight,split)['source_feasibility']['tasks'][task]['excess_over_allowance'] for s in rules['seeds']]
                        x=k+(t-1)*.22
                        ax.scatter([x-.04,x,x+.04],[v if finite(v) else np.nan for v in vs],s=10,color=color,alpha=.6)
                        if all(finite(v) for v in vs):
                            xx.append(x);yy.append(statistics.mean(vs))
                            ax.errorbar(x,statistics.mean(vs),yerr=statistics.stdev(vs),fmt='D',color=color,markersize=4,capsize=2)
                    ax.scatter([],[],color=color,label=task)
                ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),[short[a] for a in ARMS],rotation=35,ha='right')
                ax.set(title=f'{split}; '+('PWGTP' if weight=='person_weighted' else 'unweighted'),ylabel='Probe loss − PCA32 loss − .01 (nats)');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{interface}: independent source-probe feasibility, validation and development separately')
        save(fig,'source_feasibility_'+interface,'Below zero passes that task floor. Small points: all three seeds; diamonds/error bars: descriptive mean/SD. Validation is the probe-selection pool.')
    label_positions={'G_J':(.16,.97),'G_L025':(.50,.97),'G_L20':(.84,.97),'U_J':(.16,.03),'U_L025':(.50,.03),'U_L20':(.84,.03),'T':(.01,.55),'I':(.99,.45)}
    for target in ('SEX','RAC1P'):
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for i,interface in enumerate(rules['interfaces']):
            for j,weight in enumerate(rules['weights']):
                ax=axes[i,j]
                for arm in ARMS:
                    records=[point(s,interface,arm,weight) for s in rules['seeds']]
                    xs=[r['utility']['same_residence'] for r in records];ys=[r['gains']['AB/'+target] for r in records]
                    for x,y,r in zip(xs,ys,records):
                        if finite(x) and finite(y):ax.scatter(x,y,s=24,color=colors[arm],alpha=.6,marker='o' if r['source_feasibility']['pass'] is True else 'x')
                    if all(finite(v) for v in (*xs,*ys)):
                        mx,my=statistics.mean(xs),statistics.mean(ys)
                        ax.scatter(mx,my,marker='D' if arm in NEW_ARMS else 's',s=28,color=colors[arm])
                        xtext,ytext=label_positions[arm]
                        ax.annotate(f"{short[arm]} ({sum(r['source_feasibility']['pass'] is True for r in records)}/3)",(mx,my),xytext=(xtext,ytext),textcoords='axes fraction',
                            fontsize=7,color=colors[arm],ha='left' if arm=='T' else 'right' if arm=='I' else 'center',va='center',
                            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1},arrowprops={'arrowstyle':'-','lw':.4,'color':colors[arm]})
                ax.set(title=interface+'; '+('PWGTP' if weight=='person_weighted' else 'unweighted'),xlabel='Residence probe loss (nats)',ylabel=f'AB {target} gain (nats)')
                ax.margins(x=.13,y=.18);ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.grid(alpha=.2)
        fig.suptitle(f'Development evaluation: {target}, expanded catch-up, 360 epochs')
        save(fig,'tradeoff_'+target,'Small points: seeds (circle=source floors pass; ×=fail). Mean labels show source-pass n/3. These two-axis projections are not full-vector dominance.'+('\nRace code 4 is absent from independent fitting and validation; full-category assessment remains unassessable.' if target=='RAC1P' else ''))
    for interface,weight in itertools.product(rules['interfaces'],rules['weights']):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=[];utilities=[];gains=[]
        for arm,original in rules['guarded_unguarded_pairs']:
            for task in UTILITY_TASKS:
                labels.append(short[arm]+' '+task.replace('civilian_at_work','employment').replace('income_binary','income').replace('public_coverage','coverage').replace('same_residence','residence').replace('commute_over20','commute'))
                utilities.append([difference(point(s,interface,arm,weight)['utility'][task],point(s,interface,original,weight)['utility'][task]) for s in rules['seeds']])
        values_bar(axes[0],utilities,labels)
        axes[0].set(title='Guarded − same unguarded: all five utilities',ylabel='Probe-loss difference (nats)');axes[0].tick_params(axis='x',labelsize=7)
        labels=[]
        for arm,original in rules['guarded_unguarded_pairs']:
            for role in ('A/SEX','A/RAC1P','B/SEX','B/RAC1P','AB/SEX','AB/RAC1P'):
                labels.append(short[arm]+' '+role);gains.append([difference(point(s,interface,arm,weight)['gains'][role],point(s,interface,original,weight)['gains'][role]) for s in rules['seeds']])
        values_bar(axes[1],gains,labels);axes[1].set(title='Guarded − same unguarded: attribute recovery',ylabel='Signed gain difference (nats)');axes[1].tick_params(axis='x',labelsize=7)
        fig.suptitle(f'{interface}; {weight}; development evaluation')
        save(fig,'guarded_minus_unguarded_'+interface+'_'+weight,'Negative values favor guarded utility or lower recovery. Every seed is shown; error bars are descriptive three-seed SD. Race support remains incomplete.')
    for weight in rules['weights']:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,task in zip(axes,('same_residence','commute_over20')):
            values=[[difference(point(s,'F',arm,weight)['utility'][task],point(s,'P',arm,weight)['utility'][task]) for s in rules['seeds']] for arm in NEW_ARMS]
            values_bar(ax,values,[short[a] for a in NEW_ARMS],[colors[a] for a in NEW_ARMS]);ax.axhline(-.01,color='#2166ac',ls='--',label='−.01 task-advantage reference');ax.set(title=task,ylabel='F minus P probe loss (nats)');ax.legend(fontsize=8)
        fig.suptitle('Development evaluation: residence and commute capability, '+weight)
        save(fig,'feature_minus_prediction_'+weight,'These are separate task comparisons. Source floors and each attribute’s .005 extra-gain reference must still be checked. Error bars: descriptive three-seed SD.')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,interface in enumerate(rules['interfaces']):
        for j,weight in enumerate(rules['weights']):
            ax=axes[i,j];labels=[];values=[]
            for scope in SCOPES:
                for arm in GUARDED_ARMS:
                    labels.append(short[arm]+' '+scope.replace('standard_independent','standard').replace('expanded_independent','expanded').replace('expanded_catchup','pooled'))
                    values.append([point(s,interface,arm,weight,scope=scope)['gains']['AB/SEX'] for s in rules['seeds']])
            values_bar(ax,values,labels);ax.set(title=interface+'; '+weight,ylabel='AB SEX gain (nats)');ax.tick_params(axis='x',labelsize=7)
    fig.suptitle('Development evaluation: guarded systems across audit scopes, 360 epochs')
    save(fig,'audit_scopes','Three-seed means, descriptive SD and every seed. Expanded candidate pools constrain validation selection, not development ordering.')
    for target in ('SEX','RAC1P'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for i,interface in enumerate(rules['interfaces']):
            for j,weight in enumerate(rules['weights']):
                ax=axes[i,j];labels=[];values=[]
                for arm in NEW_ARMS:
                    for view in ('A','B','AB'):
                        labels.append(short[arm]+' '+view);values.append([point(s,interface,arm,weight)['gains'][view+'/'+target] for s in rules['seeds']])
                values_bar(ax,values,labels);ax.set(title=interface+'; '+weight,ylabel=target+' signed gain (nats)');ax.tick_params(axis='x',labelsize=7)
        fig.suptitle('Individual and coalition recovery: '+target+', pooled360')
        save(fig,'individual_coalition_'+target,'Development losses need not be monotone across candidate pools. All seed points and descriptive SD are shown; all nine race categories remain reported.')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,interface in enumerate(rules['interfaces']):
        for j,weight in enumerate(rules['weights']):
            ax=axes[i,j]
            for k,local in enumerate(('G_L025','G_L20')):
                rows=primary_rows(pairs,rules,interface=interface,weighting=weight)
                row=next(r for r in rows if r['local_arm']==local)
                ax.bar([k*3,k*3+1],[row['close_source_and_utility_eligible_seed_count'],row['close_qualifying_seed_count']],color=['#4393c3','#b2182b'])
                ax.text(k*3,.1,'eligible',rotation=90,ha='center',fontsize=8);ax.text(k*3+1,.1,'lower SEX',rotation=90,ha='center',fontsize=8)
            ax.set_xticks([.5,3.5],['G-J vs G-L025','G-J vs G-L20']);ax.set_ylim(0,3.3);ax.set_yticks([0,1,2,3]);ax.set(title=interface+'; '+weight,ylabel='Number of fixed-pair seeds (out of3)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Primary source-plus-residence panel: close eligibility and lower SEX, δ=.001')
    save(fig,'utility_matching','Both systems must pass each original PCA32 source floor. Counts concern the same fixed pair across seeds; no comparator stitching or mean-only qualification.')
    return [str(path.relative_to(out)) for path in sorted(folder.glob('*.png'))]


def training_evidence(out,evidence,entries):
    counts,epochs,details,native_changes=[],[],[],[]
    counted_aliases=set()
    for entry in entries:
        checkpoint=ROOT/entry['training_checkpoint']
        path=checkpoint.parent/'training.json'
        if not path.exists():path=checkpoint.parent.parent/'training.json'
        if not path.exists():continue
        record=evidence.read(path)
        metadata=record['arms'][entry['original_condition']] if 'arms' in record else record
        curve=metadata.get('curve',[]);initial=curve[0].get('counts',{}) if curve else {}
        final=metadata.get('counts',{})
        new_forward=final.get('mapper_optimizer_steps',0)-initial.get('mapper_optimizer_steps',0) if not entry['reused'] else 0
        new_observers=final.get('adversary_optimizer_steps',0)-initial.get('adversary_optimizer_steps',0) if not entry['reused'] else 0
        alias=entry.get('forward_alias',entry['condition'])
        count_forward=not entry['reused'] and alias not in counted_aliases
        if count_forward:counted_aliases.add(alias)
        counts.append({k:entry[k] for k in ('seed','condition','interface','arm','reused')} | dict(
            training_record=str(path.relative_to(ROOT)),training_record_sha256=sha(path),forward_alias=alias,
            unique_forward_counted_here=count_forward,new_interface_forward_steps=new_forward,
            new_unique_forward_steps=new_forward if count_forward else 0,
            new_live_forward_parameter_transitions=new_forward if count_forward else 0,
            new_guarded_parameter_transitions=new_forward if not entry['reused'] and entry['arm'] in GUARDED_ARMS else 0,
            new_disposable_candidate_adam_calculations=2*new_forward if not entry['reused'] and entry['arm'] in GUARDED_ARMS else 0,
            new_live_source_only_adam_step_calls=new_forward if count_forward and entry['arm']=='T' else 0,
            T_interface_forward_alias_not_a_second_transition=entry['arm']=='T' and not count_forward,
            new_observer_steps=new_observers,
            inherited_source_warmup_epochs=60,inherited_observer_warmup_passes=20,
            new_interface_source_epochs=80 if not entry['reused'] else 0,new_unique_forward_epochs=80 if count_forward else 0,
            new_observer_passes=240 if not entry['reused'] else 0,source_exposure_per_row=140,observer_exposure_per_row=260,
            shared_T=metadata.get('shared_source_only'),source_valid_label_exposures=metadata.get('source_valid_label_exposures'),
            observer_valid_label_exposures=metadata.get('observer_valid_label_exposures')))
        if entry['reused']:continue
        for row in metadata.get('epoch_update_summaries',[]):
            epochs.append({k:entry[k] for k in ('seed','condition','interface','arm')} | row)
        for row in metadata.get('detailed_steps',[]):
            item={k:entry[k] for k in ('seed','condition','interface','arm')} | row
            details.append(item)
            losses=row.get('native_losses')
            if losses:
                for task in SOURCE_TASKS:
                    values={key:losses[key]['tasks'][task] for key in ('pre','source','full','accepted')}
                    native_changes.append({k:entry[k] for k in ('seed','condition','interface','arm')} | dict(epoch=row['epoch'],task=task,
                        support=losses['pre']['support'][task],**{key+'_loss':value for key,value in values.items()},
                        source_minus_pre=difference(values['source'],values['pre']),full_minus_source=difference(values['full'],values['source']),
                        accepted_minus_source=difference(values['accepted'],values['source']),accepted_minus_full=difference(values['accepted'],values['full']),
                        accepted_minus_pre=difference(values['accepted'],values['pre']),source_proposal_has_inherited_full_moments=True))
    save_csv(out/'TRAINING_COUNTS.csv',counts);save_csv(out/'GUARD_EPOCHS.csv',epochs)
    save_json(out/'GUARD_DIAGNOSTICS.json',details);save_csv(out/'GUARD_NATIVE_CHANGES.csv',native_changes)
    return counts,epochs,details,native_changes


def guard_report(out,epochs,details,native_changes):
    records=[r for r in epochs if not r.get('literal_source_only',False)]
    summary=[]
    for condition in sorted(set(r['condition'] for r in records)):
        rows=[r for r in records if r['condition']==condition]
        total=sum(r['mapper_steps'] for r in rows)
        empty=sum(r['active_subset_counts'].get('empty',0) for r in rows)
        max_raw=[max(r['raw_dots']['max'][t] for r in rows) for t in range(3)]
        max_ideal=[max(r['ideal_dots']['max'][t] for r in rows) for t in range(3)]
        max_actual=[max(r['actual_dots']['max'][t] for r in rows) for t in range(3)]
        max_excess=[max(r['actual_dot_minus_bound']['max'][t] for r in rows) for t in range(3)]
        summary.append(dict(condition=condition,present_seeds=sorted(set(r['seed'] for r in rows)),actual_guard_steps=total,
            nonempty_active_subset_steps=total-empty,empty_active_subset_steps=empty,
            maximum_raw_task_dots=max_raw,maximum_ideal_task_dots=max_ideal,maximum_actual_task_dots=max_actual,
            maximum_actual_dot_minus_declared_bound=max_excess,task_order=list(SOURCE_TASKS),
            maximum_cast_error_l2=max(r['cast_error_l2']['max'] for r in rows),
            outside_support_identity_all_steps=all(r['outside_support_proposals_identical_all_steps'] for r in rows)))
    save_json(out/'GUARD_SUMMARY.json',{'conditions':summary,'task_order':list(SOURCE_TASKS),
        'ideal_float64_feasibility_is_not_stored_float32_1e12_feasibility':True,
        'native_source_proposal_has_inherited_protection_moments':True,'native_loss_checks_do_not_affect_updates':True})
    lines=['# What the guard changed locally','',
        'The two Adam proposals start from the same current parameter and optimizer state. The source-only current proposal retains past protection moments; it is not the trajectory T would have taken. The guard projects the current full-minus-source proposal increment on the original F/P protection support. Full-objective moments are retained once. Each guard transition performs two disposable candidate Adam calculations and one live parameter transition; T performs one ordinary live Adam step. Candidate calculations are not additional live training steps. Shared T forward transitions are counted once per seed, while both interface observer sets receive their own updates.','',
        'These are local fitting diagnostics. They do not establish monotone native loss, independent probe utility, transfer or privacy. Native losses at pre/source/full/accepted states use only the predeclared first minibatches of epochs 1/20/40/60/80; they never choose or alter an optimizer update. T uses its literal source-only path and has no guard projection. Its original step stream records source losses, support and schedules but not source-displacement norms/dots; any bounded source-only reconstruction supplement is separate evidence and must pass its own replay before being treated as complete.','',
        'Guard-support dot fields exclude the ordinary source-head displacement for F; explicitly named full-displacement dot fields include it. The source/full/accepted native losses evaluate the complete forward model on the same fitting minibatch. Ideal float64 feasibility and stored float32 task dots are separate. Positive stored dots can be explained by the declared cast and summation bound; no corrective float32 step is taken. The bound is checked against every real guard update, with full vectors saved at the diagnostic points for independent replay.','']
    rows=[]
    for r in summary:
        rows.append([r['condition'],r['present_seeds'],r['actual_guard_steps'],r['nonempty_active_subset_steps'],
            f"{max(r['maximum_ideal_task_dots']):.3g}",f"{max(r['maximum_actual_task_dots']):.3g}",
            f"{max(r['maximum_actual_dot_minus_declared_bound']):.3g}",r['outside_support_identity_all_steps']])
    lines.extend([historical.markdown_table(['Condition','Present seeds','Guard steps','Nonempty active subset','Max ideal dot','Max stored dot','Max stored dot minus bound','Outside-support identity'],rows),'',
        '[Per-epoch summaries](GUARD_EPOCHS.csv.gz) include active-set/rank counts, raw/projected/cast norms and per-task before/after dots. [Fixed-point diagnostics](GUARD_DIAGNOSTICS.json.gz) retain all compact state hashes, losses, bounds and support. [Native finite-step changes](GUARD_NATIVE_CHANGES.csv) and [guard summary](GUARD_SUMMARY.json) report signs and magnitudes without turning a fitting calculation into a downstream success criterion.',''])
    (out/'GUARD_ANALYSIS.md').write_text('\n'.join(lines))
    return summary


def t_reconstruction_report(out,evidence):
    path=out/'t_source_reconstruction/REPLAY.json'
    if not path.exists():return
    record=evidence.read(path)
    if not record.get('passed'):return
    rows=[]
    for seed,result in record['seeds'].items():
        stream=ROOT/result['step_diagnostics_path']
        if sha(stream)!=result['step_diagnostics_sha256']:
            raise ValueError('Supplemental T diagnostic stream changed')
        evidence.inputs.add(stream)
        for point in result['diagnostic_steps']:
            for i,task in enumerate(SOURCE_TASKS):
                values={stage:point['native_losses'][stage]['tasks'][task] for stage in ('pre','source','full','accepted')}
                rows.append(dict(seed=int(seed),forward_alias=f'seed_{seed}/T_shared',epoch=point['epoch'],task=task,
                    reconstructed_diagnostic=True,original_forward_fit_replaced=False,
                    source_displacement_l2=point['source_displacement_l2'],full_source_displacement_dot=point['full_source_displacement_dots'][i],
                    mapper_only_source_displacement_dot=point['mapper_only_source_displacement_dots'][i],
                    **{stage+'_loss':value for stage,value in values.items()},accepted_minus_pre=difference(values['accepted'],values['pre']),
                    accepted_minus_source=difference(values['accepted'],values['source']),full_minus_source=difference(values['full'],values['source'])))
    save_csv(out/'T_SOURCE_DIAGNOSTICS.csv',rows)
    lines=['', '## Separate exact source-only diagnostic reconstruction','',
        f"The bounded supplement passed for all {record['source_forward_trajectories']} shared T trajectories / {record['total_steps']:,} replay updates. It reproduces every original source-loss record, all five saved pre/post model and Adam states, and both final F/P model and Adam states exactly. These are verification updates, not additional fitted systems or live training transitions; the full process is separately charged to the scientific ceiling. No observer/auditor was refitted, no reserved label was read and no original evidence or release was replaced.", '',
        'The original T stream omitted displacement norms/dots. Their values now come from the separately frozen reconstruction, with a distinct provenance; they are not retroactively attributed to the original fitting logs. T has zero protection increment and no active projection. Its source/full/accepted diagnostic states are identical aliases of one ordinary Adam proposal. Individual finite-step native losses can still increase under the weighted source objective.', '',
        '[Exact replay certificate and five-point losses](t_source_reconstruction/REPLAY.json) · [Per-seed/task diagnostic rows](T_SOURCE_DIAGNOSTICS.csv) · [Prospective supplement plan](T_DIAGNOSTIC_RECONSTRUCTION_PLAN.json)', '']
    report=out/'GUARD_ANALYSIS.md';report.write_text(report.read_text()+'\n'.join(lines))


def guard_plots(out,epochs,native_changes):
    if not epochs:return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={'G_J':'#b2182b','G_L025':'#2166ac','G_L20':'#4393c3'}
    def save(fig,name,caption):
        fig.text(.5,.008,caption,ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(folder/(name+'.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,interface in enumerate(('F','P')):
        for arm,color in colors.items():
            rows=[r for r in epochs if r['interface']==interface and r['arm']==arm]
            if not rows:continue
            order=sorted(set(r['epoch'] for r in rows))
            for ax,field in zip(axes[i],('raw_l2','ideal_l2')):
                values=[[r[field]['mean'] for r in rows if r['epoch']==epoch] for epoch in order]
                for seed in (0,1,2):
                    path=[r for r in rows if r['seed']==seed]
                    ax.plot([r['epoch'] for r in path],[r[field]['mean'] for r in path],color=color,alpha=.25,lw=.7)
                ax.plot(order,[statistics.mean(v) for v in values],color=color,label=arm)
                ax.set(title=f'{interface}: epoch mean {field}',xlabel='Continuation epoch',ylabel='Parameter-displacement L2');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Current protection increment before and after the projection')
    save(fig,'guard_displacement','Faint curves: each seed; bold: mean of available seeds. These are epoch-average fitting displacements, not downstream utility or privacy guarantees.')
    if native_changes:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for i,interface in enumerate(('F','P')):
            for j,task in enumerate(SOURCE_TASKS):
                ax=axes[i,j]
                for arm,color in colors.items():
                    rows=[r for r in native_changes if r['interface']==interface and r['arm']==arm and r['task']==task]
                    for row in rows:ax.scatter(row['epoch'],row['accepted_minus_source'],color=color,s=18,alpha=.6)
                    ax.scatter([],[],color=color,label=arm)
                ax.axhline(0,color='gray',lw=.8);ax.set(title=interface+' '+task,xlabel='Predeclared diagnostic epoch',ylabel='Accepted − current source proposal BCE');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Finite-step native fitting losses: actual accepted update versus source proposal')
        save(fig,'guard_native_finite_steps','The source proposal inherits the same protection moments. These fixed minibatch outcomes are diagnostics; no acceptance test or correction uses them.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,interface in zip(axes,('F','P')):
        for arm,color in colors.items():
            rows=[r for r in epochs if r['interface']==interface and r['arm']==arm]
            for row in rows:ax.scatter(row['epoch'],max(row['actual_dot_minus_bound']['max']),s=5,color=color,alpha=.3)
            ax.scatter([],[],color=color,label=arm)
        ax.axhline(0,color='black',lw=.8);ax.set(title=interface,xlabel='Continuation epoch',ylabel='Max stored task dot − declared bound');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Stored float32 guard increments versus the declared cast/summation bound')
    save(fig,'guard_cast_bounds','Each point is one seed/epoch maximum over its guard updates and three tasks. A bounded cast residual is not a downstream-utility guarantee.')
    return [str(path.relative_to(out)) for path in sorted(folder.glob('guard_*.png'))]



def audit_budget_plot(out,points,aliases,rules):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    colors=('#1b9e77','#7570b3','#d95f02')
    roles=('AB/SEX','AB/RAC1P','B/RAC1P')
    for i,interface in enumerate(rules['interfaces']):
        for j,weight in enumerate(rules['weights']):
            ax=axes[i,j]
            for k,scope in enumerate(rules['audit_scopes']):
                for n,(arm,role) in enumerate(itertools.product(NEW_ARMS,roles)):
                    condition=aliases[0,interface,arm]['condition']
                    values=[difference(points[seed,condition,'development_evaluation',weight,scope,360]['gains'][role],
                        points[seed,condition,'development_evaluation',weight,scope,120]['gains'][role]) for seed in rules['seeds']]
                    x=n+(k-1)*.22
                    if all(finite(v) for v in values):
                        ax.bar(x,statistics.mean(values),width=.2,color=colors[k],alpha=.7,yerr=statistics.stdev(values),capsize=1)
                    ax.scatter([x-.025,x,x+.025],[v if finite(v) else float('nan') for v in values],color=colors[k],s=9,alpha=.7)
                ax.scatter([],[],color=colors[k],label=scope)
            ax.set_xticks(range(12),[arm+' '+role for arm,role in itertools.product(NEW_ARMS,roles)],rotation=45,ha='right')
            ax.axhline(0,color='black',lw=.8);ax.set(title=interface+'; '+weight,ylabel='Selected gain360 − gain120 (nats)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7);ax.tick_params(axis='x',labelsize=7)
    fig.suptitle('Nested audit-budget sensitivity: coalition SEX/race and B race')
    fig.text(.5,.008,'Development evaluation; every seed, mean and descriptive SD. All eleven roles remain in AUDIT_BUDGET.csv. Validation selection can improve while development loss worsens.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95));path=out/'figures/audit_budgets.png';fig.savefig(path,dpi=180);plt.close(fig)
    return str(path.relative_to(out))

COMPRESSED_EXPORTS=(*previous.COMPRESSED_EXPORTS,'GUARD_EPOCHS.csv','GUARD_DIAGNOSTICS.json')


def summarize(out):
    started=time.perf_counter();executing_sha=sha(Path(__file__))
    out=Path(out).resolve()
    evidence,config,rules,entries,aliases=load_evidence(out)
    index=historical.build_index(evidence.rows)
    history,context=historical.load_context(config,evidence)
    points=points_from_index(index,entries,rules)
    cube=pair_cube(points,aliases,rules);aggregates=aggregate_pairs(cube,rules)
    vector=vector_tradeoffs(cube,rules['comparison_roundoff_tolerance'])
    pareto=pareto_rows(points,entries,rules)
    metrics=previous.metric_rows(index,entries,rules);metric_aggregates=historical.aggregate_metrics(metrics)
    contrasts=fixed_contrasts(index,aliases,rules)
    fields=('comparison','interface','left_arm','right_arm','left','right','role','view','target','audit_budget','scope','evaluation_split','weighting','raw_score_split')
    contrast_aggregates=historical.aggregate_rows(contrasts,fields)
    for row in contrast_aggregates:
        row['signed_gain_difference_mean']=-row['mean'] if row['role']=='audit' and finite(row['mean']) else None
    utility,policies,feature,support=criteria_rows(index,history,context,entries,aliases,rules)
    budgets,scopes,singletons=previous.audit_changes(index,entries,rules)
    references,contextual=previous.contextual_rows(index,context,entries,rules)
    context_fields=('comparison','left','right','role','view','target','audit_budget','scope','historical_scope','historical_actual_audit_budget','split','causal_replacement')
    contextual_aggregates=historical.aggregate_rows(contextual,context_fields)
    for row in contextual_aggregates:
        row['signed_gain_difference_mean']=-row['mean'] if row['role']=='audit' and finite(row['mean']) else None
    historical.contextual_report(out,references)
    candidate_counts=historical.export_candidates(out,evidence)
    counts,epochs,details,native_changes=training_evidence(out,evidence,entries)
    guard_summary=guard_report(out,epochs,details,native_changes)
    t_reconstruction_report(out,evidence)
    native=[r for r in metrics if r['role']=='native'];native_aggregate=[r for r in metric_aggregates if r['role']=='native']
    exports=[('PER_SEED.csv',metrics),('AGGREGATE.csv',metric_aggregates),('UTILITY_MATCHES.csv',cube),('UTILITY_MATCHES_AGGREGATE.csv',aggregates),
        ('PAIRED.csv',contrasts),('PAIRED_AGGREGATE.csv',contrast_aggregates),('NONDOMINATED_POINTS.csv',pareto),('VECTOR_TRADEOFF.csv',vector),
        ('SUPPORT.csv',support),('AUDIT_BUDGET.csv',budgets),('AUDIT_SCOPE.csv',scopes),('COALITION_MINUS_SINGLETON.csv',singletons),
        ('CONTEXTUAL.csv',references),('CONTEXTUAL_PAIRED.csv',contextual),('CONTEXTUAL_PAIRED_AGGREGATE.csv',contextual_aggregates),
        ('NATIVE_SOURCE.csv',native),('NATIVE_AGGREGATE.csv',native_aggregate)]
    for name,rows in exports:save_csv(out/name,rows)
    save_json(out/'criteria.json',dict(margins=historical.MARGINS,fixed_parent='E_pca',parent_actual_audit_budget=120,
        per_seed_split_utility=utility,per_seed_view_budget_scope_policy=policies,opposing_task_threshold=None,all_target_certificate=False))
    save_json(out/'feature_capability.json',dict(margins=historical.MARGINS,tasks_decided_separately=True,per_seed=feature))
    save_json(out/'SOURCE_FEASIBILITY.json',feasibility_rows(points,rules))
    complete=len(evidence.completed_systems)==48
    table_report(out,points,entries,rules,aggregates,complete)
    matching_report(out,cube,aggregates,rules);native_report(out,index,entries,rules);individual_report(out,points,entries,rules);controls_report(out,index,rules);per_seed_reports(out,index,points,entries,rules)
    figures=plots(out,points,entries,aliases,rules,aggregates,metrics,details)
    guard_plots(out,epochs,native_changes)
    audit_budget_plot(out,points,aliases,rules)
    figures=[str(path.relative_to(out)) for path in sorted((out/'figures').glob('*.png'))]
    compressed={name:previous.gzip_export(out/name) for name in COMPRESSED_EXPORTS if (out/name).exists()}
    for name in ('TABLE.md','MATCHING_ANALYSIS.md','NATIVE_SOURCE.md','INDIVIDUAL_TARGETS.md','GUARD_ANALYSIS.md','CONTEXTUAL_REFERENCES.md','STATIC_CONTROLS.md'):
        path=out/name;text=path.read_text()
        for filename in COMPRESSED_EXPORTS:text=text.replace(']('+filename+')',']('+filename+'.gz)')
        path.write_text(text)
    save_json(out/'COMPACT_EXPORTS.json',dict(format='deterministic gzip level9, mtime0, empty filename; no archive bundle',
        decompress="gzip.decompress(Path('UTILITY_MATCHES.csv.gz').read_bytes())",files=compressed))
    main=[]
    for entry,split,weight in itertools.product((e for e in entries if e['seed']==0),rules['evaluation_splits'],rules['weights']):
        rows=[points[s,entry['condition'],split,weight,'expanded_catchup',360] for s in rules['seeds']]
        main.append(dict(condition=entry['condition'],arm=entry['arm'],interface=entry['interface'],evaluation_split=split,weighting=weight,
            source_pass_per_seed={r['seed']:r['source_feasibility']['pass'] for r in rows},
            utility={t:historical.statistics(r['utility'][t] for r in rows) for t in UTILITY_TASKS},
            gains={role:historical.statistics(r['gains'][role] for r in rows) for role in ('AB/SEX','AB/RAC1P','B/RAC1P')}))
    sources=[Path(__file__),ROOT/'scripts/summarize_acs_coalition_strength.py',ROOT/'scripts/acs_coalition_strength_comparisons.py',
        ROOT/'scripts/summarize_acs_coalition.py',ROOT/'scripts/summarize_acs_protection.py',ROOT/'scripts/summarize_acs_pca16_init.py',
        ROOT/'scripts/summarize_acs_restricted.py',ROOT/'scripts/summarize_acs_selective.py',ROOT/'scripts/summarize_acs_preservation.py']
    if sha(Path(__file__))!=executing_sha:
        raise RuntimeError('Reporter source changed during execution; rerender for publication')
    result=dict(complete=complete,evaluation_status=config['evaluation_status'],physical_systems=48,new_interface_systems=24,
        new_unique_forward_continuations=21,reused_systems=24,completed_systems=[dict(seed=s,condition=c) for s,c in sorted(evidence.completed_systems)],
        new_completed_systems=sum((e['seed'],e['condition']) in evidence.completed_systems and not e['reused'] for e in entries),
        pair_cube_rows=len(cube),fixed_pair_aggregate_rows=len(aggregates),pareto_rows=len(pareto),vector_rows=len(vector),candidate_counts=candidate_counts,
        main_conditions=main,primary_fixed_pairs=primary_rows(aggregates,rules),guard_summary=guard_summary,
        report_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
        input_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted(evidence.inputs)},
        compact_exports=compressed,figures=figures,reporting_runtime_seconds=time.perf_counter()-started)
    save_json(out/'summary.json',result)
    print({k:result[k] for k in ('complete','new_completed_systems','pair_cube_rows','fixed_pair_aggregate_rows','reporting_runtime_seconds')})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    summarize(parser.parse_args().out)
