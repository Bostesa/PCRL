"""Read-only selective-teacher study reporting, with explicit audit budgets.

Historical rows retain their original scores and fitting metadata. Canonical
names are report aliases, never claims that older runs used new source code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, SPLITS, MARGINS, TASK_DISPLAY,
    get_value, candidate_metadata, audit_coverage, combined_coverage,
    parent_comparison, feature_bank_comparison, joint_status,
    finite, statistics, number, status, table, read_json, save_json, save_csv, sha,
)
from scripts.summarize_acs_bottleneck import make_index as original_index
from scripts.summarize_acs_preservation import flatten as export_candidates, mean_sd
from scripts.summarize_acs_pca16_init import utility_criteria

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = ('R', 'E', 'S')
RHOS = ('0p1', '0')
PREFIXES = tuple(f'{t}_rho{r}' for t in TEACHERS for r in RHOS)
FINAL = tuple(f'{p}_{a}' for p in PREFIXES for a in ('C', 'D'))
NEW_FINAL = tuple(r for r in FINAL if not r.startswith('R_rho0p1_'))
STAGES = ('I', *(p+'_W' for p in PREFIXES))
DIRECT = ('PCA16', 'E_direct', 'S_direct')
BANKS = ('B_rich_bank', 'B_rich_bank_leace', 'C_tree_bank', 'C_tree_bank_leace')
REFERENCES = ('C_init', 'D_init', 'beta1_C_warmup_only', 'beta1_D_warmup_only',
              'E_pca', 'E_pca_leace', 'C_bottleneck', 'D_protected', *BANKS, 'prior')
RELEASES = (*FINAL, *DIRECT, *REFERENCES, *STAGES)
LABELS = {'PCA16': 'R: raw PCA16', 'E_direct': 'E: real-attribute teacher', 'S_direct': 'S: permuted-label teacher',
    'C_init': 'beta0 C_init', 'D_init': 'beta0 D_init', 'I': 'I: raw-PCA16 initialization',
    'beta1_C_warmup_only': 'Historical beta1 warmup-only C', 'beta1_D_warmup_only': 'Historical beta1 warmup-only D',
    'E_pca': 'Original PCA32', 'E_pca_leace': 'PCA32 + LEACE', 'C_bottleneck': 'Historical C16',
    'D_protected': 'Historical D16', 'B_rich_bank': 'Rich neural bank', 'C_tree_bank': 'Rich tree bank',
    'B_rich_bank_leace': 'Neural bank + LEACE', 'C_tree_bank_leace': 'Tree bank + LEACE',
    'prior': 'Fitting prior', 'exposed': 'Exposed control'}
for _prefix in PREFIXES:
    _t, _rho = _prefix.split('_rho')
    for _stage in ('W', 'C', 'D'):
        LABELS[_prefix+'_'+_stage] = f'{_t}, rho={".1" if _rho == "0p1" else "0"}, {_stage}'


def rename_key(key, rename):
    role, release, target = key.split('/')
    return '/'.join((role, rename(release), target))


def index_records(records):
    index = original_index(records)
    for seed, role, release, target in sorted({key[:4] for key in index}):
        if role == 'audit' and (release in STAGES or release.endswith('_W')):
            raise ValueError('I/W are not independently audited and cannot inherit teacher audit scores')
        selected = 'independent_selected' if role == 'audit' else 'selected'
        row = index.get((seed, role, release, target, selected))
        if row is not None:
            index[seed, role, release, target, 'primary'] = row
        if role == 'audit':
            row = index.get((seed, role, release, target, 'selected'))
            if row is not None:
                index[seed, role, release, target, 'pooled'] = row
    return index


class Evidence:
    """Deduplicate saved rows while keeping budget-specific metadata and origins."""
    def __init__(self):
        self.rows = {120: {}, 360: {}}
        self.meta = {120: {}, 360: {}}
        self.inputs = set()
        self.units = []

    def read(self, path):
        path = Path(path).resolve()
        self.inputs.add(path)
        return read_json(path)

    def add(self, raw, metadata, budget, origin, rename=lambda x: x, *, replace=False, reused=True):
        row = copy.deepcopy(raw)
        original_name = row['release']
        row['release'] = rename(original_name)
        role = row['role']
        key = (row['seed'], role, row['release'], row['target'], row['candidate_id'])
        row['reused_reference'] = reused
        row['evidence_audit_budget'] = raw.get('audit_budget', 120) if role == 'audit' else None
        row['report_audit_budget'] = budget if role == 'audit' else None
        row['origin_metrics'] = str(Path(origin).relative_to(ROOT))
        row['origin_release'] = original_name
        previous = self.rows[budget].get(key)
        if previous is not None and not replace:
            fields = (*SPLITS, 'selected', 'independent_selected', 'candidate_id', 'family')
            if any(previous.get(k) != row.get(k) for k in fields):
                raise ValueError('Conflicting reused prediction/selection evidence: '+str(key))
            return
        self.rows[budget][key] = row
        self.meta[budget][key] = copy.deepcopy(metadata)

    def records(self, budget, *, utility=True):
        seeds = sorted({key[0] for key in self.rows[budget]})
        return [{'seed': seed, 'raw_metrics': [row for key, row in self.rows[budget].items()
                 if key[0] == seed and (utility or row['role'] == 'audit')]} for seed in seeds]

    def selections(self, budget):
        result = defaultdict(lambda: {'fitting_records': {}})
        for key, metadata in self.meta[budget].items():
            seed, role, release, target, candidate = key
            result[seed]['fitting_records'].setdefault(f'{role}/{release}/{target}', {'candidates': {}})['candidates'][candidate] = metadata
        return dict(result)


def historical_alias(release, *, extended=False):
    aliases = {'W': 'R_rho0p1_W', 'C_persistent': 'R_rho0p1_C', 'D_persistent': 'R_rho0p1_D',
               'C_warmup_only': 'beta1_C_warmup_only', 'D_warmup_only': 'beta1_D_warmup_only'}
    if extended:
        aliases = {'beta_1_C_persistent': 'R_rho0p1_C', 'beta_1_D_persistent': 'R_rho0p1_D',
                   'beta_1_C_warmup_only': 'beta1_C_warmup_only', 'beta_1_D_warmup_only': 'beta1_D_warmup_only'}
    return aliases.get(release, release)


def load_evidence(out):
    evidence = Evidence()
    config = evidence.read(out/'config.json')
    evidence.inputs.add(out/'PROTOCOL.md')
    parent = (ROOT/config['preservation_reference_results']).resolve()
    allowed = {*RELEASES, 'exposed'}
    for seed in config['seeds']:
        path = parent/'beta_1'/f'seed_{seed}'/'metrics.json'
        record = evidence.read(path)
        selected = evidence.read(path.parent/'selection_before_test.json')
        for raw in record['raw_metrics']:
            if historical_alias(raw['release']) not in allowed:
                continue
            metadata = candidate_metadata(selected, raw)
            for budget in (120, 360):
                evidence.add(raw, metadata, budget, path, historical_alias)
        path = parent/'extended'/f'seed_{seed}'/'metrics.json'
        record = evidence.read(path)
        selected = evidence.read(path.parent/'selection_before_test.json')
        for raw in record['raw_metrics']:
            if historical_alias(raw['release'], extended=True) not in allowed:
                continue
            budget = raw['audit_budget']
            evidence.add(raw, candidate_metadata(selected['budgets'][str(budget)], raw), budget, path,
                lambda r: historical_alias(r, extended=True), replace=True)
    for prefix in ('static', *(p for p in PREFIXES if p != 'R_rho0p1')):
        for seed in config['seeds']:
            path = out/prefix/f'seed_{seed}'/'metrics.json'
            if not path.exists():
                continue
            record = evidence.read(path)
            if record['seed'] != seed:
                raise ValueError('Seed identity differs from completed unit path')
            selected = evidence.read(path.parent/'selection_before_test.json')
            rename = lambda r: f'{prefix}_{r}' if prefix != 'static' and r in ('W', 'C', 'D') else r
            for raw in record['raw_metrics']:
                if raw['role'] == 'audit':
                    budget = raw['audit_budget']
                    if budget not in (120, 360):
                        raise ValueError('Undeclared audit budget')
                    metadata = candidate_metadata(selected['budgets'][str(budget)], raw)
                    evidence.add(raw, metadata, budget, path, rename, reused=raw['release'] == 'I')
                else:
                    metadata = candidate_metadata(selected, raw)
                    for budget in (120, 360):
                        evidence.add(raw, metadata, budget, path, rename, reused=raw['release'] == 'I')
            evidence.units.append({'unit': prefix, 'seed': seed, 'metrics': str(path.relative_to(out)),
                                   'runtime': record.get('runtime'), 'integrity': record.get('integrity')})
    return evidence, config


def comparison_matrix():
    pairs = []
    for rho in RHOS:
        for arm in ('C', 'D'):
            pairs += [('E_minus_R', f'E_rho{rho}_{arm}', f'R_rho{rho}_{arm}'),
                      ('E_minus_S', f'E_rho{rho}_{arm}', f'S_rho{rho}_{arm}')]
    for teacher in TEACHERS:
        for arm in ('C', 'D'):
            pairs.append(('rho0_minus_rho0p1', f'{teacher}_rho0_{arm}', f'{teacher}_rho0p1_{arm}'))
    for prefix in PREFIXES:
        pairs += [('D_minus_C', prefix+'_D', prefix+'_C'), ('W_minus_I', prefix+'_W', 'I')]
        for arm in ('C', 'D'):
            release = prefix+'_'+arm
            pairs.append(('final_minus_W', release, prefix+'_W'))
            if release in NEW_FINAL:
                pairs.append(('new_final_minus_beta0', release, arm+'_init'))
            if prefix[0] in ('E', 'S'):
                pairs.append(('learned_minus_own_teacher', release, prefix[0]+'_direct'))
    return pairs


def paired_rows(seeds, indices):
    rows = []
    for seed in seeds:
        for comparison, left, right in comparison_matrix():
            for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
                if role == 'audit' and (left in STAGES or right in STAGES):
                    continue
                for budget in ((120,) if role == 'transfer' else (120, 360)):
                    for selector in (('primary',) if role == 'transfer' else ('primary', 'pooled')):
                        for target in targets:
                            for raw_split, split in SPLITS.items():
                                index = indices[budget]
                                a = index.get((seed, role, left, target, selector))
                                b = index.get((seed, role, right, target, selector))
                                av = a.get(raw_split, {}).get('log_loss') if a else None
                                bv = b.get(raw_split, {}).get('log_loss') if b else None
                                rows.append({'seed': seed, 'comparison': comparison, 'left': left, 'right': right,
                                    'role': role, 'target': target, 'audit_budget': budget if role == 'audit' else None,
                                    'selector': selector, 'split': split, 'left_value': av, 'right_value': bv,
                                    'left_candidate': a['candidate_id'] if a else None,
                                    'right_candidate': b['candidate_id'] if b else None,
                                    'left_evidence_audit_budget': a.get('evidence_audit_budget') if a else None,
                                    'right_evidence_audit_budget': b.get('evidence_audit_budget') if b else None,
                                    'left_minus_right': av-bv if finite(av) and finite(bv) else None,
                                    'signed_gain_left_minus_right': bv-av if role == 'audit' and finite(av) and finite(bv) else None})
    return rows


def aggregate_rows(rows, fields, value='left_minus_right'):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return [{**dict(zip(fields, key)), **statistics(r[value] for r in group),
             'per_seed': {str(r['seed']): r[value] for r in group}} for key, group in groups.items()]


def interaction_rows(pairs):
    """Difference of predeclared reconstruction contrasts; no arm selection."""
    fields = ('seed', 'left', 'role', 'target', 'audit_budget', 'selector', 'split')
    indexed = {tuple(row[k] for k in fields): row for row in pairs if row['comparison'] == 'rho0_minus_rho0p1'}
    rows = []
    for pair in pairs:
        if pair['comparison'] != 'rho0_minus_rho0p1' or not pair['left'].startswith('E_'):
            continue
        for reference in ('R', 'S'):
            key = tuple(reference+pair[k][1:] if k == 'left' else pair[k] for k in fields)
            other = indexed[key]
            a, b = pair['left_minus_right'], other['left_minus_right']
            defined = finite(a) and finite(b)
            rows.append({**{k: pair[k] for k in fields if k != 'left'}, 'reference_teacher': reference,
                'arm': pair['left'][-1], 'comparison': 'E_reconstruction_interaction_vs_'+reference,
                'E_rho0': pair['left'], 'E_rho0p1': pair['right'],
                'reference_rho0': other['left'], 'reference_rho0p1': other['right'],
                'E_reconstruction_delta': a, 'reference_reconstruction_delta': b,
                'left_minus_right': a-b if defined else None,
                'signed_gain_interaction': b-a if pair['role'] == 'audit' and defined else None})
    return rows


def aggregate_metrics(seeds, indices):
    rows = []
    for release in RELEASES:
        for role, targets in (('transfer', UTILITY_TASKS), ('audit', ATTRIBUTES)):
            if role == 'audit' and release in STAGES:
                continue
            for budget in ((120,) if role == 'transfer' else (120, 360)):
                for selector in (('primary',) if role == 'transfer' else ('primary', 'pooled')):
                    for target in targets:
                        for raw_split, split in SPLITS.items():
                            values = [get_value(indices[budget], s, role, release, target, raw_split, selector=selector) for s in seeds]
                            meta = {'release': release, 'role': role, 'target': target, 'audit_budget': budget if role == 'audit' else None,
                                    'selector': selector, 'split': split}
                            rows.append({**meta, 'metric': 'log_loss', **statistics(values)})
                            if role == 'audit':
                                priors = [get_value(indices[budget], s, role, 'prior', target, raw_split, selector='primary') for s in seeds]
                                gains = [p-v if finite(p) and finite(v) else None for p, v in zip(priors, values)]
                                rows.append({**meta, 'metric': 'signed_prior_relative_gain', **statistics(gains)})
    return rows


def criteria(seeds, indices, selections):
    utility_rows, policies, banks, coverage = [], [], [], []
    for seed in seeds:
        for raw_split, split in SPLITS.items():
            utility = lambda r: {t: get_value(indices[120], seed, 'transfer', r, t, raw_split, selector='primary') for t in UTILITY_TASKS}
            bank_residence = {b: utility(b)['same_residence'] for b in ('B_rich_bank', 'C_tree_bank')}
            for release in (*FINAL, *DIRECT, *STAGES, *REFERENCES):
                utility_rows.append({'seed': seed, 'split': split, 'release': release,
                    **utility_criteria(utility(release), utility('E_pca'), bank_residence)})
            for budget in (120, 360):
                for selector in ('primary', 'pooled'):
                    index = indices[budget]
                    attack = lambda r: {t: get_value(index, seed, 'audit', r, t, raw_split, selector=selector) for t in ATTRIBUTES}
                    cov = lambda r, t: audit_coverage(index, selections[budget][seed], seed, r, t, raw_split, selector)
                    for release in (*FINAL, *DIRECT):
                        attr_cov = {t: combined_coverage(cov(release, t), cov('E_pca', t)) for t in ATTRIBUTES}
                        common = {'seed': seed, 'split': split, 'release': release, 'audit_budget': budget, 'audit_selector': selector}
                        policies.append({**common, 'fixed_parent': 'E_pca', 'parent_actual_audit_budget': 120,
                            **parent_comparison(utility('E_pca'), utility(release), bank_residence,
                                attack('E_pca'), attack(release), attack('prior'), attr_cov)})
                        coverage.extend({**common, 'target': t, **cov(release, t)} for t in ATTRIBUTES)
                        if release not in FINAL:
                            continue
                        for bank in BANKS:
                            comparison = feature_bank_comparison(utility(release)['same_residence'], utility(bank)['same_residence'],
                                attack(release), attack(bank), attack('prior'),
                                {t: combined_coverage(cov(release, t), cov(bank, t)) for t in ATTRIBUTES})
                            banks.append({**common, 'bank': bank, 'numeric_joint_inequalities': joint_status([
                                comparison['utility']['pass'], *[v['numeric_inequality'] for v in comparison['attributes'].values()]]), **comparison})
    return utility_rows, policies, banks, coverage


def export_all(out, evidence):
    exports = [[], [], [], []]
    for budget in (120, 360):
        records = evidence.records(budget, utility=budget == 120)
        batches = export_candidates(records, evidence.selections(budget), budget)
        for destination, source in zip(exports, batches):
            for row in source:
                key = tuple(row[k] for k in ('seed', 'role', 'release', 'target', 'candidate_id'))
                raw = evidence.rows[budget][key]
                row.update({k: raw[k] for k in ('evidence_audit_budget', 'origin_metrics', 'origin_release')})
            destination.extend(source)
    for name, rows in zip(('PER_TARGET.csv', 'PER_CLASS.csv', 'FITTING.csv', 'CURVES.csv'), exports):
        save_csv(out/name, rows)
    return exports


def mechanism_rows(out, evidence):
    scores, coordinates, direct, gradients, training, teachers = [], [], [], [], [], []
    for unit in evidence.units:
        directory = out/unit['unit']/f"seed_{unit['seed']}"
        teacher_path = directory/'teachers/teachers.json'
        if teacher_path.exists():
            record = evidence.read(teacher_path)
            for teacher, diagnostic in record['diagnostics'].items():
                teachers.append({'seed': unit['seed'], 'teacher': teacher, **diagnostic,
                    'map_metadata': record['maps'].get(teacher), 'permutation': record['permutation'],
                    'original_scale': record['original_scale'], 'original_mean': record['original_mean']})
        path = directory/'geometry.json'
        if path.exists():
            record = evidence.read(path)
            for name, snapshot in record['snapshots'].items():
                release = unit['unit']+'_'+name if unit['unit'] != 'static' and name in ('W', 'C', 'D') else name
                for target, target_record in snapshot['targets'].items():
                    common = {'unit': unit['unit'], 'seed': unit['seed'], 'release': release,
                        'teacher': snapshot.get('teacher'), 'target': target,
                        **{k: v for k, v in target_record.items()
                           if k not in ('fit', 'source_validation', 'development_evaluation')}}
                    for split in ('fit', 'source_validation', 'development_evaluation'):
                        score = target_record.get(split)
                        if score is None:
                            continue
                        scores.append({**common, 'split': split, **{k: v for k, v in score.items() if not isinstance(v, list)}})
                        for metric, values in score.items():
                            if isinstance(values, list):
                                coordinate_identity = {k: common[k] for k in ('unit', 'seed', 'release', 'teacher', 'target')}
                                coordinates.extend({**coordinate_identity, 'split': split, 'metric': metric, 'coordinate': j, 'value': value}
                                                   for j, value in enumerate(values))
                for split, targets in snapshot.get('direct_teacher_error', {}).items():
                    direct.extend({'unit': unit['unit'], 'seed': unit['seed'], 'release': release,
                                   'split': split, 'target': target, **score} for target, score in targets.items())
        path = directory/'training/training.json'
        if path.exists():
            record = evidence.read(path)
            for point, values in record.get('stage_gradient_diagnostics', {}).items():
                scope = 'D-objective hypothetical at frozen fork; no shared-stage update' if point == 'shared_fork' else 'shared-stage diagnostic'
                gradients.append({'unit': unit['unit'], 'seed': unit['seed'], 'stage': 'shared', 'point': point,
                                  'diagnostic_scope': scope, **values})
            for stage, meta in record['arms'].items():
                for point in ('shared_fork', 'final'):
                    gradients.append({'unit': unit['unit'], 'seed': unit['seed'], 'stage': stage, 'point': point,
                                      **meta.get('fixed_batch_gradient_diagnostics_at_'+point, {})})
                training.extend({'unit': unit['unit'], 'seed': unit['seed'], 'stage': stage, **r} for r in meta['curve'])
            for stage, points in record.get('shared_curves', {}).items():
                training.extend({'unit': unit['unit'], 'seed': unit['seed'], 'stage': stage, **r} for r in points)
    for name, rows in (('MECHANISM.csv', scores), ('MECHANISM_COORDINATES.csv', coordinates),
            ('DIRECT_MATCHING.csv', direct), ('GRADIENT_DIAGNOSTICS.csv', gradients), ('TRAINING_CURVES.csv', training),
            ('TEACHER_GEOMETRY.csv', teachers)):
        save_csv(out/name, rows)
    return {'scores': scores, 'direct': direct, 'gradients': gradients, 'teachers': teachers}


def audit_budget_rows(seeds, indices):
    rows, catchup = [], []
    for seed in seeds:
        for release in (*FINAL, *DIRECT, 'C_init', 'D_init', 'beta1_C_warmup_only', 'beta1_D_warmup_only', 'B_rich_bank', 'C_tree_bank'):
            for target in ATTRIBUTES:
                for raw_split, split in SPLITS.items():
                    for selector in ('primary', 'pooled', *('candidate:'+c for c in
                            ('logistic', 'mlp_0', 'mlp_1', 'hist_gb_20', 'hist_gb_5', 'catchup', 'saved_adversary'))):
                        losses = [get_value(indices[b], seed, 'audit', release, target, raw_split, selector=selector) for b in (120, 360)]
                        rows.append({'seed': seed, 'release': release, 'target': target, 'split': split, 'selector': selector,
                            'loss_120': losses[0], 'loss_360': losses[1],
                            'loss_360_minus_120': losses[1]-losses[0] if all(finite(v) for v in losses) else None})
                    for budget in (120, 360):
                        result = {'seed': seed, 'release': release, 'target': target, 'split': split, 'audit_budget': budget}
                        for name, selector in (('saved', 'candidate:saved_adversary'), ('fresh', 'primary'),
                                               ('catchup', 'candidate:catchup'), ('pooled', 'pooled')):
                            raw = indices[budget].get((seed, 'audit', release, target, selector))
                            result[name+'_loss'] = raw.get(raw_split, {}).get('log_loss') if raw else None
                            result[name+'_candidate'] = raw['candidate_id'] if raw else None
                        catchup.append(result)
    return rows, catchup


def audit_change_rows(seeds, indices, policies, banks):
    """Numerical margin/rank changes retain support-gated statuses separately."""
    parent_rows, bank_rows, rank_rows = [], [], []
    grouped = defaultdict(dict)
    for row in policies:
        grouped[tuple(row[k] for k in ('seed', 'split', 'release', 'audit_selector'))][row['audit_budget']] = row
    for key, budgets in grouped.items():
        for target in ATTRIBUTES:
            before, after = [budgets[b]['attribute_halving'][target] for b in (120, 360)]
            parent_rows.append({**dict(zip(('seed', 'split', 'release', 'selector'), key)), 'target': target,
                'numeric_120': before['numeric_halving_inequality'], 'numeric_360': after['numeric_halving_inequality'],
                'pass_120': before['pass'], 'pass_360': after['pass'],
                'numeric_changed': before['numeric_halving_inequality'] != after['numeric_halving_inequality'],
                'assessment_changed': before['pass'] != after['pass'], 'parent_actual_audit_budget': 120})
    grouped = defaultdict(dict)
    for row in banks:
        grouped[tuple(row[k] for k in ('seed', 'split', 'release', 'audit_selector', 'bank'))][row['audit_budget']] = row
    for key, budgets in grouped.items():
        before, after = [budgets[b] for b in (120, 360)]
        bank_rows.append({**dict(zip(('seed', 'split', 'release', 'selector', 'bank'), key)),
            'numeric_120': before['numeric_joint_inequalities'], 'numeric_360': after['numeric_joint_inequalities'],
            'numeric_changed': before['numeric_joint_inequalities'] != after['numeric_joint_inequalities']})
    releases = (*FINAL, *DIRECT, 'C_init', 'D_init', 'beta1_C_warmup_only', 'beta1_D_warmup_only', 'B_rich_bank', 'C_tree_bank')
    for seed in seeds:
        for target in ATTRIBUTES:
            for raw_split, split in SPLITS.items():
                for selector in ('primary', 'pooled'):
                    losses = {budget: {r: get_value(indices[budget], seed, 'audit', r, target, raw_split, selector=selector)
                                       for r in releases} for budget in (120, 360)}
                    for release in releases:
                        ranks = {}
                        for budget in (120, 360):
                            value = losses[budget][release]
                            ranks[budget] = 1 + sum(v < value for v in losses[budget].values() if finite(v)) if finite(value) else None
                        rank_rows.append({'seed': seed, 'target': target, 'split': split, 'selector': selector, 'release': release,
                            'rank_120': ranks[120], 'rank_360': ranks[360], 'rank_changed': ranks[120] != ranks[360],
                            'loss_120': losses[120][release], 'loss_360': losses[360][release],
                            'rank_direction': 'lower loss = more recovery; numerical descriptive rank, tied values share rank'})
    return parent_rows, bank_rows, rank_rows


def write_audit_report(out, evidence, budget, catchup, changes, coverage):
    parent, banks, ranks = changes
    curves = []
    for key, metadata in evidence.meta[360].items():
        seed, role, release, target, candidate = key
        if role != 'audit' or candidate not in ('mlp_0', 'mlp_1', 'catchup'):
            continue
        trajectory = metadata.get('validation_curve') or []
        if not trajectory:
            continue
        last = trajectory[-1].get('validation_log_loss')
        best = min(p['validation_log_loss'] for p in trajectory)
        curves.append({'seed': seed, 'release': release, 'target': target, 'candidate': candidate,
            'new_fitting': not evidence.rows[360][key]['reused_reference'],
            'selected_epoch': metadata.get('selected_epoch'), 'last_epoch': trajectory[-1]['epoch'],
            'last_minus_best_validation_loss': last-best if finite(last) else None})
    save_csv(out/'AUDIT_SATURATION.csv', curves)
    new_curves = [r for r in curves if r['new_fitting']]
    new_catchup = [r for r in new_curves if r['candidate'] == 'catchup']
    epoch0 = [r for r in new_catchup if r['selected_epoch'] == 0]
    epoch0_pooled = [r for r in epoch0 if evidence.rows[360][
        (r['seed'], 'audit', r['release'], r['target'], 'catchup')]['selected']]
    selected_rows = [r for r in budget if r['selector'] in ('primary', 'pooled') and finite(r['loss_360_minus_120'])]
    count_changed = lambda rows, key: sum(r[key] is True for r in rows)
    lines = ['# Audit budget and inherited-exposure sensitivity', '',
        'Fresh MLP and learned catch-up checkpoints at120/360 are nested within the SAME360-epoch '
        'trajectory. Logistic/tree candidates do not change. The separate saved_adversary candidate '
        'is diagnostic and excluded from selection. Its identical epoch0 predictions remain eligible '
        'inside the catch-up trajectory under the inherited checkpoint rule. '
        'Catch-up starts from its own final observer in direct coordinates with reset Adam; its '
        'representation-fitting exposure prevents calling it matched-independent. Static teachers '
        'have neither observer nor catch-up. Every selection uses unweighted validation log loss, '
        'and PWGTP evaluates the same selected predictions.', '',
        f'{len(epoch0)}/{len(new_catchup)} new catch-up trajectories select epoch0; '
        f'{len(epoch0_pooled)} also win pooled selection: '
        + ', '.join(f'seed{r["seed"]} {r["target"]} {LABELS[r["release"]]}' for r in epoch0_pooled)+'. '
        'These selections reproduce the saved observer before any catch-up update. A pooled-versus-fresh '
        'gap can therefore reflect inherited representation-fitting exposure without beneficial '
        'additional optimization. The gap is not automatically evidence that longer attacker fitting helped.', '',
        '[Every candidate and selection delta](AUDIT_BUDGET.csv), [saved/fresh/catch-up](CATCHUP.csv), '
        '[curves](CURVES.csv), [checkpoint saturation](AUDIT_SATURATION.csv), '
        '[parent numeric changes](AUDIT_PARENT_CHANGES.csv), [bank changes](AUDIT_BANK_CHANGES.csv), '
        '[every release rank](AUDIT_RANKING.csv).', '',
        'Original PCA32/LEACE, erased banks and historical C16/D16 keep their original120 evidence. '
        'Original PCA32 remains the fixed descriptive parent at both budgets. Matched360 rankings '
        'below exclude those120-only references.', '',
        f'{sum(abs(r["loss_360_minus_120"]) > 1e-12 for r in selected_rows)}/{len(selected_rows)} '
        'defined seed/target/split/scope selected-loss comparisons change by more than1e-12.', '',
        f'Numeric parent attribute inequalities change in {count_changed(parent, "numeric_changed")}/{len(parent)} '
        f'comparisons; coverage-gated assessments change in {count_changed(parent, "assessment_changed")}/{len(parent)}. '
        f'Joint bank inequalities change in {count_changed(banks, "numeric_changed")}/{len(banks)}. '
        f'Numerical recovery ranks change in {count_changed(ranks, "rank_changed")}/{len(ranks)} '
        'seed/target/split/scope/release records. Repeated scopes are descriptive records, not independent observations.', '',
        '## Selected checkpoints and remaining scope', '',
        f'{sum((r["selected_epoch"] or 0) > 120 for r in new_curves)}/{len(new_curves)} newly fitted '
        'noncontrol trajectories select a checkpoint after120. Historical controls and references are '
        'separately marked in AUDIT_SATURATION.csv. Positive last-minus-best loss means the final '
        'iterate has worse validation loss than its saved best checkpoint.', '',
        'Unchanged nested selections support saturation only for these fixed trajectories and candidate '
        'families. Changed or late checkpoints indicate budget sensitivity; neither result establishes '
        'privacy, rules out stronger recovery, or repairs absent category support. No extension beyond360 was run.', '']
    (out/'AUDIT_BUDGET.md').write_text('\n'.join(lines))
    supports = [r for r in coverage if r['audit_budget'] == 360 and r['audit_selector'] == 'primary'
                and r['split'] == 'development_evaluation']
    lines = ['# Protected-label support and exposed controls', '',
        'All nine RAC1P Census codes remain in every score. Code4 lacks attacker fitting and validation '
        'support in this fixed cohort; additional fitting cannot repair it. Full race conclusions remain '
        'unassessable. Aggregate gains stay signed and visible. A numerical inequality is distinct from '
        'a supported assessment. The exposed-label control is evaluated with the matching nested budget.', '',
        '[Complete per-class scores including exposed/prior controls](PER_CLASS.csv), '
        '[all budgets/scopes/support flags](SUPPORT.csv).', '']
    lines += table(['Seed', 'Release', 'Attribute', 'Coverage complete', 'Census-code limitations', 'Exposed candidate'],
        [[r['seed'], LABELS[r['release']], r['target'], r['complete'], r['limitation_census_codes'], r['exposed_candidate']] for r in supports])
    (out/'SUPPORT.md').write_text('\n'.join(lines))
    return {'selected_score_comparisons': len(selected_rows),
        'selected_score_changes': sum(abs(r['loss_360_minus_120']) > 1e-12 for r in selected_rows),
        'new_trajectories': len(new_curves), 'new_selected_after120': sum((r['selected_epoch'] or 0) > 120 for r in new_curves),
        'parent_numeric_changes': count_changed(parent, 'numeric_changed'),
        'bank_numeric_changes': count_changed(banks, 'numeric_changed'), 'ranking_changes': count_changed(ranks, 'rank_changed')}


def write_reports(out, seeds, aggregates, pairs, utility, policies, banks, geometry, units, interactions):
    ai = {(r['release'], r['role'], r['target'], r['audit_budget'], r['selector'], r['split'], r['metric']): r for r in aggregates}
    def stat(release, role, target, split, budget=360, selector='primary', metric='log_loss'):
        return ai.get((release, role, target, budget if role == 'audit' else None, selector, split, metric),
                      {'complete': False, 'n_defined': 0, 'n_seeds': len(seeds), 'mean': None, 'sample_sd': None})
    avg = lambda *args, **kwargs: mean_sd(stat(*args, **kwargs))
    def paired_cell(release, role, target, selector='primary', metric='log_loss'):
        return ' / '.join(number(stat(release, role, target, split, selector=selector, metric=metric)['mean'])
                          for split in ('development_evaluation', 'development_evaluation_person_weighted'))
    intro = 'DEVELOPMENT EVALUATION on the original households and cohort. Unweighted loss is primary; '
    intro += 'PWGTP uses the SAME unweighted-validation-selected predictions. Lower task losses and signed '
    intro += 'prior-relative attack gains are better under the named finite audit. Three shared-cohort seed SDs are descriptive.'
    lines = ['# Selective preservation: primary comparison', '', intro, '',
        f'Completed new units: {len(units)}/18 (three static-teacher units and fifteen learned C/D units). '
        'Six R/rho.1 learned releases are reused. Missing units remain undefined, not favorable subset averages.', '',
        '[Decision](RESEARCH_DECISION.md), [analysis](ANALYSIS.md), [teachers](TEACHERS.md), [mechanism](MECHANISM.md), '
        '[all candidates](PER_TARGET.csv), [every class/control](PER_CLASS.csv), [paired contrasts](PAIRED.csv).', '',
        'Plots: [SEX independent](tradeoff_SEX.png), [SEX pooled](tradeoff_SEX_pooled.png), '
        '[RAC1P independent](tradeoff_RAC1P.png), [RAC1P pooled](tradeoff_RAC1P_pooled.png), '
        '[all five stage utilities](utility_stages.png).', '',
        '## Development means: 360-epoch independent audits', '',
        'The main learned/direct rows and beta0 references below all have nested360 evidence. '
        'Named original PCA32/LEACE, erased-bank and C16/D16 references later retain historical120 '
        'audits; original PCA32 remains the fixed mixed-budget parent.', '',
        'Cells show unweighted / PWGTP. Source counts require all three source losses individually '
        'within original PCA32+.01. RAC1P retains all nine classes but is support-limited.', '']
    primary = []
    for release in (*FINAL, *DIRECT, 'C_init', 'D_init'):
        row = {'release': release}
        count = []
        for split in ('development_evaluation', 'development_evaluation_person_weighted'):
            criteria_for_release = [r for r in utility if r['release'] == release and r['split'] == split]
            count.append(f"{sum(r['source_preservation']['pass'] is True for r in criteria_for_release)}/{len(seeds)}")
        cells = [LABELS[release], ' / '.join(count), paired_cell(release, 'transfer', 'same_residence'),
                 *[paired_cell(release, 'audit', t, metric='signed_prior_relative_gain') for t in ATTRIBUTES]]
        primary.append(cells)
    lines += table(['Release', 'Source all U / W', 'Residence U / W', 'SEX gain U / W', 'RAC1P gain U / W'], primary)
    lines += ['', '## Catch-up-inclusive 360 sensitivity', '',
        'The separate saved_adversary candidate is excluded from selection. Its identical epoch0 '
        'predictions remain eligible inside catch-up. Catch-up inherits representation-fitting exposure '
        'and preserves direct release coordinates; its pooled advantage need not come from additional '
        'optimizer updates. [The three epoch0 pooled winners](AUDIT_BUDGET.md) are disclosed separately. '
        'Static teachers have no catch-up.', '']
    lines += table(['Release', 'SEX gain U / W', 'RAC1P gain U / W'],
        [[LABELS[r], *[paired_cell(r, 'audit', t, 'pooled', 'signed_prior_relative_gain') for t in ATTRIBUTES]] for r in FINAL])
    def criterion_cell(release, selector, fetch):
        values = []
        for split in ('development_evaluation', 'development_evaluation_person_weighted'):
            flags = [fetch(r) for r in policies if r['release'] == release and r['audit_budget'] == 360
                     and r['audit_selector'] == selector and r['split'] == split]
            values.append(f'{sum(flag is True for flag in flags)}/{len(seeds)}'
                          + (f' ({sum(flag is None for flag in flags)} undefined)' if any(flag is None for flag in flags) else ''))
        return ' / '.join(values)
    lines += ['', '## Unchanged descriptive margins: seed pass counts', '',
        'Unweighted / PWGTP; fixed original PCA32 parent retains120 audit evidence. Residence '
        'requires half positive parent headroom; SEX requires half positive parent attack gain. '
        'RAC1P numeric inequalities below are NOT supported assessments: every full race assessment '
        'is unassessable because fixed-category support is incomplete.', '']
    lines += table(['Release', 'Residence retention U / W', 'SEX half independent U / W',
                    'SEX half pooled U / W', 'RAC1P numeric half pooled U / W'],
        [[LABELS[r], criterion_cell(r, 'primary', lambda p: p['residential_retention']['pass']),
          *[criterion_cell(r, selector, lambda p: p['attribute_halving']['SEX']['pass']) for selector in ('primary', 'pooled')],
          criterion_cell(r, 'pooled', lambda p: p['attribute_halving']['RAC1P']['numeric_halving_inequality'])] for r in FINAL])
    for split in SPLITS.values():
        lines += ['', f"## {split.replace('_', ' ')}: all five utility losses and independent-360 audits", '']
        lines += table(['Release', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX loss', 'RAC1P loss'],
            [[LABELS[r], *[avg(r, 'transfer', t, split) for t in UTILITY_TASKS],
              *['not audited' if r in STAGES else avg(r, 'audit', t, split) for t in ATTRIBUTES]] for r in RELEASES])
    lines += ['', 'Original PCA32, PCA32+LEACE, erased banks and historical C16/D16 retain their historical '
        '120-epoch evidence even in a table headed 360. The actual candidate budget and source path are '
        'explicit in PER_TARGET.csv. Original PCA32 remains the fixed parent; unprotected banks have matched360 evidence.', '']
    (out/'TABLE.md').write_text('\n'.join(lines))
    analysis = ['# Selective preservation: prespecified contrasts', '', intro, '',
        'E−R compares true-label teacher erasure with raw teacher preservation. E−S compares real-attribute '
        'erasure with one fixed paired-label permutation; any realized rank/distortion mismatch would limit that control. '
        'rho0−rho.1 intervenes on reconstruction of original PCA32. D−C changes only the real protection '
        'gradient within a full-state fork. E-C still depends on real attributes through its fixed teacher.', '',
        'Positive task-loss contrasts lose utility; positive attack-loss contrasts reduce measured recovery. '
        'Neither teacher construction nor any release is selected using these outcomes.', '',
        'The standalone saved_adversary candidate is excluded, but the same predictions at epoch0 '
        'remain eligible within catch-up. Three new seed1 SEX trajectories (E/rho.1/D, E/rho0/C '
        'and E/rho0/D) select epoch0 and win pooled selection. Thus a pooled/fresh reversal can '
        'reflect inherited representation-fitting exposure without useful additional optimization. '
        '[Audit-budget evidence](AUDIT_BUDGET.md) separates this from any effect of longer fitting.', '',
        '[Every paired seed score](PAIRED.csv), [all paired aggregates](PAIRED_AGGREGATE.csv), '
        '[unchanged criteria](criteria.json), [bank comparisons](bank_comparisons.json), '
        '[budget sensitivity](AUDIT_BUDGET.csv), [saved/fresh/catch-up](CATCHUP.csv).', '']
    pair_index = {(r['comparison'], r['left'], r['right'], r['target'], r['audit_budget'], r['selector'], r['split']): r for r in pairs}
    def contrast_cell(comparison, left, right, target, selector='primary'):
        budget = 360 if target in ATTRIBUTES else None
        return ' / '.join(number(pair_index.get((comparison, left, right, target, budget, selector, split), {}).get('mean'))
            for split in ('development_evaluation', 'development_evaluation_person_weighted'))
    if len(units) == 18:
        raw_pairs = [r for r in pairs if r['comparison'] == 'E_minus_R' and r['target'] == 'same_residence'
                     and r['split'] == 'development_evaluation']
        sham_sex = [r for r in pairs if r['comparison'] == 'E_minus_S' and r['target'] == 'SEX'
                    and r['audit_budget'] == 360 and r['selector'] == 'pooled' and r['split'] == 'development_evaluation']
        source_ok = [r for r in utility if r['release'] in FINAL and r['split'] in
                     ('development_evaluation', 'development_evaluation_person_weighted')]
        teacher_by = {(r['seed'], r['teacher']): r for r in geometry['teachers']}
        same_geometry = all(teacher_by[s, 'E']['map_metadata']['projection_retained_rank'] ==
                            teacher_by[s, 'S']['map_metadata']['projection_retained_rank'] and
                            abs(teacher_by[s, 'E']['movement_from_raw_mean_mse']-
                                teacher_by[s, 'S']['movement_from_raw_mean_mse']) < 1e-6 for s in seeds)
        findings = ['## Findings from the complete fixed matrix', '',
            '**Real versus raw teacher.** E reduces mean measured recovery relative to R, while '
            f'its residence loss increases by {number(min(r["mean"] for r in raw_pairs))}–'
            f'{number(max(r["mean"] for r in raw_pairs))} unweighted across the four matched conditions. '
            f'All {sum(r["source_preservation"]["pass"] is True for r in source_ok)}/{len(source_ok)} '
            'final seed/weight records meet the three source-loss allowances. This is a utility/recovery '
            'tradeoff, not dominance or privacy.', '',
            '**Real versus permuted teacher.** E has MORE pooled SEX recovery than S in '
            f'{sum(r["mean"] < 0 for r in sham_sex)}/{len(sham_sex)} unweighted matched-condition means. '
            'Race differences and seed signs vary. E improves mean residence relative to S, but the '
            'attribute-specific protection interpretation is weak after catch-up. '
            + ('E/S projection ranks and original-scaled distortion match within1e-6 in every seed; '
               'the realized control is not explained by unequal compression on those two measures.' if same_geometry else
               'E/S realized rank or distortion differs; that mismatch limits the sham comparison.'), '',
            '**Reconstructing original PCA32.** Removing reconstruction from E-C changes pooled '
            'SEX/race attack LOSS by '
            + contrast_cell('rho0_minus_rho0p1', 'E_rho0_C', 'E_rho0p1_C', 'SEX', 'pooled')+' and '
            + contrast_cell('rho0_minus_rho0p1', 'E_rho0_C', 'E_rho0p1_C', 'RAC1P', 'pooled')+
            ' (unweighted / PWGTP), reducing recovery. Residence loss changes by '
            + contrast_cell('rho0_minus_rho0p1', 'E_rho0_C', 'E_rho0p1_C', 'same_residence')+
            ', and civilian-at-work loss by '
            + contrast_cell('rho0_minus_rho0p1', 'E_rho0_C', 'E_rho0p1_C', 'civilian_at_work')+
            '. R and S also respond; reconstruction pressure is not uniquely a real-erasure phenomenon.', '',
            '**Protection gradient.** E/rho.1 D−C raises pooled SEX attack loss by '
            + contrast_cell('D_minus_C', 'E_rho0p1_D', 'E_rho0p1_C', 'SEX', 'pooled')+
            ', with residence change '+contrast_cell('D_minus_C', 'E_rho0p1_D', 'E_rho0p1_C', 'same_residence')+
            '. After removing reconstruction, its SEX difference is '
            + contrast_cell('D_minus_C', 'E_rho0_D', 'E_rho0_C', 'SEX', 'pooled')+
            '; the incremental effect depends on the training objective.', '',
            '**Learned versus direct teacher and stronger audits.** All four E students improve mean '
            'utility on all five tasks under both weights over direct E, while adding recovery. '
            'For E/rho0 D, pooled SEX/race gains are '+paired_cell('E_rho0_D', 'audit', 'SEX', 'pooled', 'signed_prior_relative_gain')+
            ' and '+paired_cell('E_rho0_D', 'audit', 'RAC1P', 'pooled', 'signed_prior_relative_gain')+
            ', versus direct E '+paired_cell('E_direct', 'audit', 'SEX', 'pooled', 'signed_prior_relative_gain')+
            ' and '+paired_cell('E_direct', 'audit', 'RAC1P', 'pooled', 'signed_prior_relative_gain')+
            '. Catch-up can reverse comparisons despite unchanged120/360 checkpoints; see '
            '[budget/exposure analysis](AUDIT_BUDGET.md). Person weighting is shown beside each primary contrast.', '',
            '**Recoverable structure and design implication.** E students retain E accurately and '
            'recover part of qE that direct E cannot reconstruct affinely. This persists at rho0. '
            'Removed-component recoverability does not track protected-label gains monotonically, and '
            'qE is not pure sensitive information. The full-PCA32 input route and source objectives '
            'remain candidate paths for reintroduction. [Mechanism evidence](MECHANISM.md) supports '
            'testing that route explicitly; [the single next design](NEXT_DESIGN.md) is a proposal, '
            'not an additional run or a solution to purpose-specific combined-access coordination.', '']
        analysis[4:4] = findings
    analysis += ['## Controlled final contrasts', '',
        'Cells are mean paired loss differences, unweighted / PWGTP. Audit columns use360. '
        'For attributes, the signed attack-gain difference is the NEGATIVE of this loss difference '
        'and is also explicit in PAIRED.csv. All five tasks, every seed, validation and both budgets '
        'remain in the linked CSVs; no result is selected for this table.', '']
    analysis += table(['Left − right', 'Residence U / W', 'SEX independent U / W',
        'RAC1P independent U / W', 'SEX pooled U / W', 'RAC1P pooled U / W'],
        [[LABELS[left]+' − '+LABELS[right], contrast_cell(c, left, right, 'same_residence'),
          *[contrast_cell(c, left, right, target, selector) for selector in ('primary', 'pooled') for target in ATTRIBUTES]]
         for c, left, right in comparison_matrix() if c in ('E_minus_R', 'E_minus_S', 'rho0_minus_rho0p1', 'D_minus_C')])
    interaction_index = {(r['reference_teacher'], r['arm'], r['target'], r['audit_budget'], r['selector'], r['split']): r for r in interactions}
    def interaction_cell(reference, arm, target, selector='primary'):
        return ' / '.join(number(interaction_index.get((reference, arm, target, 360 if target in ATTRIBUTES else None,
            selector, split), {}).get('mean')) for split in ('development_evaluation', 'development_evaluation_person_weighted'))
    analysis += ['', '## Reconstruction interactions', '',
        'Each value is (E/rho0−E/rho.1)−(reference/rho0−reference/rho.1), using paired seeds. '
        'A positive attack-loss interaction means removing reconstruction reduced E recovery more '
        'than reference recovery. A positive utility interaction means a larger E utility cost. '
        'Cells are unweighted / PWGTP; full five-task, both-budget, validation and mean±SD results '
        'are in [INTERACTIONS.csv](INTERACTIONS.csv) and [INTERACTIONS_AGGREGATE.csv](INTERACTIONS_AGGREGATE.csv).', '']
    analysis += table(['Reference / arm', 'Residence U / W', 'SEX independent U / W', 'RAC1P independent U / W',
        'SEX pooled U / W', 'RAC1P pooled U / W'],
        [[f'{reference} / {arm}', interaction_cell(reference, arm, 'same_residence'),
          *[interaction_cell(reference, arm, target, selector) for selector in ('primary', 'pooled') for target in ATTRIBUTES]]
         for reference in ('R', 'S') for arm in ('C', 'D')])
    analysis += ['', '## New learned release versus its direct teacher', '',
        'Independent audits are matched by fitting pool and candidate budget. Pooled learned audits '
        'add inherited observer exposure; the static teacher has no catch-up.', '']
    analysis += table(['Learned − own teacher', *[TASK_DISPLAY[t]+' U / W' for t in UTILITY_TASKS],
        'SEX independent U / W', 'RAC1P independent U / W'],
        [[LABELS[left]+' − '+LABELS[right], *[contrast_cell(c, left, right, t) for t in (*UTILITY_TASKS, *ATTRIBUTES)]]
         for c, left, right in comparison_matrix() if c == 'learned_minus_own_teacher'])
    analysis += ['', 'Stage W−I, final−W, beta0 comparisons and all source-task costs are complete in '
        'PAIRED.csv/PAIRED_AGGREGATE.csv. Stage utility losses and source margins are shown in TABLE.md '
        'and utility_stages.png. No I/W audit is inferred from its target teacher.', '']
    analysis += ['## Margins and support', '',
        'Every source loss permits +.01 nats relative to original PCA32. Residence must retain half '
        'positive PCA32 headroom over the stronger unprotected bank. Nonpositive denominators are undefined. '
        'Attribute halving requires positive PCA32 gain and complete category/control coverage. Bank '
        'comparisons retain the .01 utility advantage and .005 maximum extra attribute gain. These '
        'are descriptive margins, not privacy budgets or noninferiority tests. No known failure is hidden '
        'by another undefined criterion. Historical helper keys containing "erased" name the compared '
        'release; they do not imply a final eraser was applied.', '',
        'All nine race codes remain. Code4 is absent from attacker fitting/validation for every seed. '
        'Missing support is unassessable; signed gains are never clipped. All support and exposed-control '
        'failures remain in PER_CLASS.csv and SUPPORT.csv. I/W receive no new attribute audit.', '']
    (out/'ANALYSIS.md').write_text('\n'.join(analysis))
    teacher_text = ['# Direct teacher evidence', '',
        'R is the original raw PCA16. E fits joint SEX2/RAC1P9 LEACE on real representation-fitting '
        'labels. S uses one frozen permutation of complete-case SEX/RAC1P pairs, with joint counts '
        'and missing masks preserved. The concatenated concept schema has11 columns, not18 intersections. '
        'All observers, D gradients and audits use real labels. No teacher is updated toward a student.', '',
        'Original raw-PCA16 fitting scales are used throughout; no erased-teacher whitening or '
        'residual-variance normalization. E/S are attribute-informed targets and contain no reserved-task labels. '
        'S need not match E in realized rank or distortion. A sham contrast cannot automatically identify '
        'attribute specificity independently of those geometric differences.', '',
        '[Full teacher geometry/maps/permutation evidence](TEACHER_GEOMETRY.csv), '
        '[direct matching errors](DIRECT_MATCHING.csv), [all teacher candidates](PER_TARGET.csv).', '']
    teacher_text += table(['Seed', 'Teacher', 'Projection retained rank', 'Movement MSE', 'Joint counts preserved', 'Masks preserved'],
        [[r['seed'], r['teacher'], (r['map_metadata'] or {}).get('projection_retained_rank', 16),
          number(r['movement_from_raw_mean_mse']), r['permutation']['joint_counts_preserved'], r['permutation']['missing_masks_preserved']]
         for r in geometry['teachers']])
    for split in ('development_evaluation', 'development_evaluation_person_weighted'):
        teacher_text += ['', f"## {split.replace('_', ' ')}", '']
        teacher_text += table(['Teacher', *[TASK_DISPLAY[t] for t in UTILITY_TASKS], 'SEX independent360', 'RAC1P independent360'],
            [[LABELS[r], *[avg(r, 'transfer', t, split) for t in UTILITY_TASKS],
              *[avg(r, 'audit', t, split) for t in ATTRIBUTES]] for r in DIRECT])
    teacher_text += ['', 'Direct teachers are independent interfaces. Their audits are not attributed to I/W '
        'or to learned students. Small fitting covariance with labels is not a nonlinear recovery guarantee.', '']
    (out/'TEACHERS.md').write_text('\n'.join(teacher_text))
    mechanism = ['# Teacher structure and local optimization diagnostics', '',
        'Separate affine decoders predict raw PCA16, retained E and removed qE=t−E from each frozen '
        'release; S conditions additionally predict retained S and qS. The removed residual is not pure '
        'sensitive information. Targets use original raw-PCA16 scales, with fitting-only centers/intercepts '
        'and fixed rcond=1e-12. Evaluation target variance is descriptive and never refits a decoder. '
        'Near-zero prior denominators (≤1e-12 in original-standardized units) give undefined ratios.', '',
        '[All mean/prior/variance/rank results](MECHANISM.csv), [every coordinate](MECHANISM_COORDINATES.csv), '
        '[direct matching](DIRECT_MATCHING.csv), [gradient norms/dots/cosines](GRADIENT_DIAGNOSTICS.csv), '
        '[objective curves and coefficients](TRAINING_CURVES.csv), '
        '[retained/removed structure plot](retained_removed_structure.png), '
        '[raw/applied gradient directions](gradient_directions.png).', '',
        'Low affine error supports recoverability; high error does not exclude nonlinear recovery. '
        'Real-attribute audits, task utility and these reconstruction diagnostics answer different questions.', '']
    mechanism += ['## Development affine errors', '',
        'Cells are three-seed mean MSE / mean per-seed MSE-to-prior ratio. Ratios are dimensionless; '
        'each target keeps its own fitting prior. Source-validation, every seed/coordinate and full '
        'spectra remain in the linked CSVs. Raw R/rho.1 geometry toward E/qE is unmeasured.', '']
    mechanism += table(['Snapshot', 'Raw t: MSE / ratio', 'Retained E: MSE / ratio', 'Removed qE: MSE / ratio',
        'Retained S: MSE / ratio', 'Removed qS: MSE / ratio'],
        [[LABELS[release], *[' / '.join(number(statistics(next((r.get(metric) for r in geometry['scores']
            if r['seed'] == seed and r['release'] == release and r['target'] == target
            and r['split'] == 'development_evaluation'), None) for seed in seeds)['mean'])
            for metric in ('mean_mse', 'mse_over_prior')) for target in ('raw', 'E', 'qE', 'S', 'qS')]]
         for release in ('I', *(p+'_W' for p in PREFIXES if p != 'R_rho0p1'), *NEW_FINAL, 'E_direct', 'S_direct')])
    mechanism += ['']
    mechanism += ['## Applied versus hypothetical gradients', '',
        'Fixed first common-base minibatch diagnostics occur at I, W, common fork and final C/D. '
        'Raw protection is positive entropy-normalized CE; D applies coefficient−.1, C zero. '
        'rho0 has zero applied reconstruction and no decoder gradients/Adam updates even when a '
        'hypothetical reconstruction gradient is reported. Teacher/reconstruction, teacher/protection '
        'and source/protection dots/cosines keep raw and applied versions distinct; zero-vector cosine '
        'is undefined. Pre-observer diagnostics use explicitly labeled disposable initialized observers.', '',
        'The common shared-fork diagnostic evaluates the prospective D objective hypothetically. '
        'No protection gradient is applied during common warmup. The arm-specific fork diagnostics '
        'record the actual C coefficient0 and D coefficient−.1 separately.', '',
        'A projection retained rank is a map-construction property. Numerical release/design ranks '
        'at rcond1e-12 may include tiny directions left by the fixed float32 release convention; '
        'full spectra are preserved. Rank alone does not measure useful or sensitive recoverability. '
        'Reused R/rho.1 releases have no new E/qE decoder fit; missing geometry remains unmeasured.', '',
        'No diagnostic takes an optimizer step or changes RNG/state. Local gradient opposition is '
        'not causal proof that an objective caused recoverability. Controlled rho contrasts provide '
        'stronger intervention evidence; reserved-task outcomes never select a diagnostic point.', '']
    (out/'MECHANISM.md').write_text('\n'.join(mechanism))


def plots(out, seeds, indices, geometry, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def value(release, role, target, split, budget=360, selector='primary'):
        return statistics(get_value(indices[budget], s, role, release, target, split, selector=selector) for s in seeds)['mean']
    shown = (*FINAL, *DIRECT, 'E_pca', 'E_pca_leace', 'B_rich_bank', 'C_tree_bank', 'C_init', 'D_init')
    for target, selector in ((t, s) for t in ATTRIBUTES for s in ('primary', 'pooled')):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
        for ax, split, title in zip(axes, ('test', 'test_person_weighted'), ('Unweighted primary', 'PWGTP: same predictions')):
            prior = value('prior', 'audit', target, split)
            for j, release in enumerate(shown):
                x, attack = value(release, 'transfer', 'same_residence', split), value(release, 'audit', target, split, selector=selector)
                if not all(finite(v) for v in (x, attack, prior)):
                    continue
                ax.scatter(x, prior-attack, s=35, marker='o' if release in FINAL else 'X',
                           color=plt.get_cmap('tab20')(j % 20), label=f'{j+1}: '+LABELS[release])
                ax.annotate(str(j+1), (x, prior-attack), xytext=(3, 4 if j % 2 else -8), textcoords='offset points', fontsize=7)
            ax.set(title=title, xlabel='Residence log loss (lower is better)', ylabel=f'{target} signed attack gain (lower is better)')
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc='lower center', fontsize=7)
        scope = 'independent360' if selector == 'primary' else 'pooled360: catch-up exposure differs'
        suffix = '' if selector == 'primary' else '_pooled'
        fig.suptitle(f'{target}: development, {scope}; historical PCA32/LEACE audit120')
        fig.tight_layout(rect=(0, .27, 1, .94)); fig.savefig(out/f'tradeoff_{target}{suffix}.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    stages = ('I', *(p+'_W' for p in PREFIXES), *FINAL)
    for ax, target in zip(axes.flat, UTILITY_TASKS):
        for split, label, marker in (('test', 'Unweighted', 'o'), ('test_person_weighted', 'PWGTP', 's')):
            vals = [value(r, 'transfer', target, split) for r in stages]
            ax.plot(range(len(stages)), [v if finite(v) else np.nan for v in vals], marker=marker, linestyle='none', label=label)
        ax.set(title=TASK_DISPLAY[target], ylabel='Log loss')
        ax.set_xticks(range(len(stages)), [r.replace('_rho', '').replace('0p1', '.1').replace('_', '') for r in stages], rotation=75, fontsize=6)
        ax.grid(axis='y', alpha=.2)
    axes.flat[-1].axis('off'); axes.flat[0].legend(fontsize=8)
    fig.suptitle('Development evaluation: task losses (nats); C/D parallel continuations from W')
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out/'utility_stages.png', dpi=180); plt.close(fig)
    if geometry['scores']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        releases = ('I', *(p+'_W' for p in PREFIXES if p != 'R_rho0p1'), *NEW_FINAL, 'E_direct', 'S_direct')
        for ax, target in zip(axes.flat, ('raw', 'E', 'qE', 'S', 'qS')):
            for metric, marker, label in (('mean_mse', 'o', 'Affine error'), ('prior_mean_mse', '^', 'Fitting prior')):
                values = [statistics(r.get(metric) for r in geometry['scores'] if r['release'] == release and r['target'] == target
                                     and r['split'] == 'development_evaluation')['mean'] for release in releases]
                ax.plot(range(len(releases)), [v if finite(v) else np.nan for v in values], marker=marker, linestyle='none', label=label)
            ax.set(title=target, ylabel='Original-standardized target MSE'); ax.set_ylim(bottom=0)
            ax.set_xticks(range(len(releases)), [r.replace('_rho', '').replace('0p1', '.1') for r in releases], rotation=75, fontsize=7)
            ax.grid(axis='y', alpha=.2)
        axes.flat[0].legend(fontsize=8); axes.flat[-1].axis('off')
        fig.suptitle('Development reconstruction of raw, retained and removed teacher structure')
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out/'retained_removed_structure.png', dpi=180); plt.close(fig)
    if geometry['gradients']:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
        prefixes = [p for p in PREFIXES if p != 'R_rho0p1']
        for i, version in enumerate(('raw', 'applied')):
            for j, pair in enumerate(('teacher_reconstruction', 'teacher_protection', 'source_protection')):
                ax = axes[i, j]
                for stage, marker in (('C', 'o'), ('D', 's')):
                    means = [statistics(r.get(f'{pair}_{version}_cosine') for r in geometry['gradients']
                        if r['unit'] == p and r['stage'] == stage and r['point'] == 'final')['mean'] for p in prefixes]
                    ax.plot(range(len(prefixes)), [v if finite(v) else np.nan for v in means], linestyle='none', marker=marker, label=stage)
                ax.set(title=pair.replace('_', ' / ')+f' ({version})', ylim=(-1.05, 1.05), ylabel='Mapper-gradient cosine')
                ax.set_xticks(range(len(prefixes)), [p.replace('_rho', ' rho ').replace('0p1', '.1') for p in prefixes], fontsize=7)
                ax.axhline(0, color='gray', linewidth=.5); ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Fixed final C/D diagnostic: raw versus coefficient-applied gradient directions\nZero-vector cosine undefined; rho0 reconstruction is hypothetical only')
        fig.tight_layout(rect=(0, 0, 1, .91)); fig.savefig(out/'gradient_directions.png', dpi=180); plt.close(fig)
    for target in ATTRIBUTES:
        fig, axes = plt.subplots(3, 4, figsize=(14, 8.5), sharex=True)
        for ax, release in zip(axes.flat, FINAL):
            for candidate in ('mlp_0', 'mlp_1', 'catchup'):
                points = defaultdict(list)
                for seed in seeds:
                    row = indices[360].get((seed, 'audit', release, target, 'candidate:'+candidate))
                    if row is None:
                        continue
                    # Fitting curves are read from exact frozen metadata; no refitting here.
                    for point in metadata[(seed, 'audit', release, target, candidate)].get('validation_curve', []):
                        points[point['epoch']].append(point['validation_log_loss'])
                if points:
                    ax.plot(sorted(points), [statistics(points[e])['mean'] for e in sorted(points)], label=candidate)
            ax.axvline(120, color='gray', linewidth=.5); ax.set(title=LABELS[release], xlabel='Epoch', ylabel='Validation log loss')
            ax.grid(alpha=.15)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{target}: same-trajectory fresh/catch-up validation curves; seed means, exposure scopes differ')
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out/f'audit_curves_{target}.png', dpi=160); plt.close(fig)


def summarize(out):
    out = Path(out).resolve()
    evidence, config = load_evidence(out)
    seeds = config['seeds']
    indices = {b: index_records(evidence.records(b)) for b in (120, 360)}
    selections = {b: evidence.selections(b) for b in (120, 360)}
    exports = export_all(out, evidence)
    aggregates = aggregate_metrics(seeds, indices)
    paired = paired_rows(seeds, indices)
    pair_summary = aggregate_rows(paired, ('comparison', 'left', 'right', 'role', 'target', 'audit_budget', 'selector', 'split'))
    interactions = interaction_rows(paired)
    interaction_summary = aggregate_rows(interactions,
        ('comparison', 'reference_teacher', 'arm', 'role', 'target', 'audit_budget', 'selector', 'split'))
    utility, policies, banks, coverage = criteria(seeds, indices, selections)
    geometry = mechanism_rows(out, evidence)
    budget, catchup = audit_budget_rows(seeds, indices)
    changes = audit_change_rows(seeds, indices, policies, banks)
    audit_summary = write_audit_report(out, evidence, budget, catchup, changes, coverage)
    for filename, rows in (('AGGREGATE.csv', aggregates), ('PAIRED.csv', paired), ('PAIRED_AGGREGATE.csv', pair_summary),
            ('SUPPORT.csv', coverage), ('AUDIT_BUDGET.csv', budget), ('CATCHUP.csv', catchup),
            ('INTERACTIONS.csv', interactions), ('INTERACTIONS_AGGREGATE.csv', interaction_summary)):
        save_csv(out/filename, rows)
    for filename, rows in zip(('AUDIT_PARENT_CHANGES.csv', 'AUDIT_BANK_CHANGES.csv', 'AUDIT_RANKING.csv'), changes):
        save_csv(out/filename, rows)
    save_json(out/'criteria.json', {'margins': MARGINS, 'fixed_parent': 'E_pca', 'parent_audit_budget': 120,
        'per_seed_split_utility': utility, 'per_seed_split_budget_policy': policies})
    save_json(out/'bank_comparisons.json', {'margins': MARGINS, 'per_seed_split_budget': banks})
    write_reports(out, seeds, aggregates, pair_summary, utility, policies, banks, geometry, evidence.units, interaction_summary)
    plots(out, seeds, indices, geometry, evidence.meta[360])
    summary = {'evaluation_status': config['evaluation_status'], 'declared_seeds': seeds,
        'new_units': evidence.units, 'complete_new_units': len(evidence.units), 'expected_new_units': 18,
        'complete': len(evidence.units) == 18, 'reused_final_conditions': ['R_rho0p1_C', 'R_rho0p1_D'],
        'aggregate_metrics': aggregates, 'paired_aggregate_metrics': pair_summary, 'interaction_aggregate_metrics': interaction_summary,
        'fixed_parent': 'E_pca', 'margins': MARGINS, 'unaudited_snapshots': list(STAGES),
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(evidence.inputs)},
        'report_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT/'scripts/summarize_acs_preservation.py', ROOT/'scripts/summarize_acs_protection.py',
             ROOT/'scripts/summarize_acs_bottleneck.py', ROOT/'scripts/summarize_acs_pca16_init.py')},
        'row_counts': dict(zip(('scores', 'classes', 'fits', 'curves'), map(len, exports))),
        'geometry_rows': len(geometry['scores'])}
    summary['audit_budget_summary'] = audit_summary
    save_json(out/'summary.json', summary)
    print({'complete_new_units': len(evidence.units), **summary['row_counts']})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    summarize(parser.parse_args().out)
