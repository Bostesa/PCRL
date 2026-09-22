"""Descriptive figures from explicitly supplied, accepted aggregate grids.

This module never discovers results, fits a model, selects a configuration, or
opens person-level artifacts. Validation input is ``VALIDATION_GRID.json`` from
reporting.py. Evaluation input has ``phase='evaluation', pool='test',
selection_frozen=True`` and the same records/receipts envelope; each record is
the native evaluation summary containing its ``test`` role dictionary.

Every supplied configuration with all three anchors is displayed. An explicit
metadata mapping supplies separately registered branch specifications. POINTS
stores the exact aggregate source pointers and accepted-summary receipts used
for every point. Recovery is a measured attacker CE reduction relative to H,
not an estimate of mutual information. The full grid has no inference tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from . import config

ANCHORS = (0, 1, 2)
ROLES = ('A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P')
WEIGHTS = {'unweighted': 'unweighted', 'PWGTP': 'weighted'}
TASK = 'utility:A/same_residence'
STAGES = ('continuous', 'code', 'unprotected_action', 'protected_L', 'protected_C')
HISTORICAL = frozenset(('J', 'leace_A0', 'splince_A0', 'optnet16_L1',
                        'optnet16_L2', 'optnet16_C1'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pointer(*parts):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts)


def _anchor(value):
    if isinstance(value, bool) or str(value) not in ('0', '1', '2'):
        raise ValueError(f'Unknown anchor: {value!r}')
    return int(value)


def _ce(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'CE must be numeric: {context}')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'CE must be finite and nonnegative: {context}')
    return float(value)


def _metadata(names, supplied):
    supplied = supplied or {}
    if not isinstance(supplied, dict):
        raise ValueError('metadata must map configuration names to specifications')
    defaults = {r['configuration']: r for r in config.release_ledger()['records']}
    result = {}
    for name in names:
        if name not in defaults and name not in supplied:
            raise ValueError(f'Unregistered configuration needs explicit metadata: {name}')
        spec = {**defaults.get(name, {}), **supplied.get(name, {})}
        canonical = spec.get('canonical_release', name)
        code = spec.get('input')
        if code is None:
            code = next((c for c in config.INPUTS if canonical.startswith(c + '_')), None)
        if code not in (*config.INPUTS, None):
            raise ValueError(f'Unknown input code for {name}')
        policy = spec.get('policy')
        if policy not in ('L', 'C', 'U', None):
            raise ValueError(f'Unknown privacy policy for {name}')
        if policy is not None and code is None:
            raise ValueError(f'Finite map requires an input code: {name}')
        budget = spec.get('budget')
        if budget is not None:
            _ce(budget, f'{name}/budget')
        stage = None
        if name == 'H':
            category = 'service_only'
        elif name in HISTORICAL:
            category = 'historical'
        elif policy in ('L', 'C'):
            category, stage = 'new_Q', 'protected_' + policy
        else:
            category = 'label_matched'
            if policy == 'U':
                stage = 'unprotected_action'
            elif canonical == 'continuous_task':
                stage = 'continuous'
            elif canonical in tuple(c + '_code' for c in config.INPUTS):
                stage = 'code'
        actions = spec.get('max_actions')
        if actions is None and (policy is not None or '_rr_' in canonical or '_withhold_' in canonical
                                or canonical in ('constant_best', 'independent_token')):
            actions = 17
        if actions is not None and (isinstance(actions, bool) or not isinstance(actions, int) or actions < 1):
            raise ValueError(f'Invalid action alphabet for {name}')
        normalized = {'configuration': name, 'category': category, 'input': code,
                      'policy': policy, 'budget': spec.get('budget'), 'actions': actions,
                      'conditioning_family': spec.get('conditioning_family', 'primary'),
                      'stage': stage, 'branch': spec.get('branch')}
        for key in ('registration_id', 'extra_registration_sha256', 'robustness_registration_sha256',
                    'matched_frontier', 'baseline_family', 'label_matched', 'canonical_release'):
            if key in spec:
                normalized[key] = spec[key]
        result[name] = normalized
    return result


def build_point_table(payload, metadata=None):
    """Validate a supplied grid and return JSON-safe, traceable plotting points.

    Receipts are provenance declarations from the accepted-grid assembler;
    this aggregate-only API cannot verify the underlying private files. It
    checks unique receipt coverage and preserves both hashes in every point.
    Missing anchors are explicitly listed and never partially averaged.
    """
    if not isinstance(payload, dict):
        raise ValueError('Aggregate grid must be an object')
    phase = payload.get('phase', 'validation' if payload.get('evaluation_opened') is False else None)
    if phase == 'validation':
        if payload.get('evaluation_opened') is not False or payload.get('pool') is not None:
            raise ValueError('Validation input must explicitly declare evaluation_opened=false')
        pool = None
        selected, independent, fixed = 'validation', 'independent_validation', 'fixed_decoder'
    elif phase == 'evaluation':
        if payload.get('selection_frozen') is not True or payload.get('pool') != 'test':
            raise ValueError('Evaluation figures require selection_frozen=true and pool=test')
        if payload.get('evaluation_opened') is False:
            raise ValueError('Mixed validation/evaluation declarations')
        pool = 'test'
        selected, independent, fixed = 'ce', 'independent_ce', 'fixed_decoder_ce'
    else:
        raise ValueError('An explicit validation or frozen evaluation phase is required')
    records = payload.get('records')
    if not isinstance(records, dict) or not records or any(not isinstance(n, str) or not n for n in records):
        raise ValueError('records must be a nonempty configuration mapping')
    specs = _metadata(records, metadata)
    receipt_map = {}
    receipts = payload.get('receipts')
    if not isinstance(receipts, list):
        raise ValueError('Accepted summary receipts are required')
    for r in receipts:
        if not isinstance(r, dict) or not isinstance(r.get('configuration'), str):
            raise ValueError('Invalid accepted summary receipt')
        key = (r['configuration'], _anchor(r.get('anchor')))
        if key in receipt_map:
            raise ValueError(f'Duplicate accepted summary receipt: {key}')
        for field in ('summary_sha256', 'receipt_sha256'):
            if not isinstance(r.get(field), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', r[field]):
                raise ValueError(f'Invalid {field}: {key}')
        receipt_map[key] = r

    adapted, unavailable = {}, {}
    for name, values in records.items():
        if not isinstance(values, dict):
            raise ValueError(f'Invalid anchor records for {name}')
        adapted[name] = {}
        for raw_anchor, native in values.items():
            anchor = _anchor(raw_anchor)
            if anchor in adapted[name]:
                raise ValueError(f'Duplicate anchor aliases for {name}')
            if (name, anchor) not in receipt_map:
                raise ValueError(f'Missing accepted summary receipt: {name}/{anchor}')
            if not isinstance(native, dict):
                raise ValueError(f'Invalid native summary for {name}/{anchor}')
            roles = native if pool is None else native.get(pool)
            if not isinstance(roles, dict):
                raise ValueError(f'Missing {phase} role dictionary: {name}/{anchor}')
            for role in (TASK, *('attack:' + r for r in ROLES)):
                entry = roles.get(role)
                if not isinstance(entry, dict) or not isinstance(entry.get(selected), dict):
                    raise ValueError(f'Missing native {phase} CE: {name}/{anchor}/{role}')
                for field in (selected, independent, fixed):
                    metrics = entry.get(field)
                    if metrics is None:
                        continue
                    if not isinstance(metrics, dict):
                        raise ValueError(f'Invalid CE metrics for {name}/{anchor}/{role}/{field}')
                    for key in WEIGHTS.values():
                        _ce(metrics.get(key), f'{name}/{anchor}/{role}/{field}/{key}')
            adapted[name][anchor] = (roles, raw_anchor)
        missing = sorted(set(ANCHORS) - set(adapted[name]))
        if missing:
            unavailable[name] = {'missing_anchors': missing, 'reason': 'Complete three-anchor mean unavailable'}
    if 'H' not in adapted or 'H' in unavailable:
        raise ValueError('A complete accepted H baseline is required for every anchor')
    complete = sorted(set(records) - set(unavailable))

    def get(name, anchor, role, field, weight):
        roles, raw_anchor = adapted[name][anchor]
        value = roles[role].get(field)
        if value is None:
            return None
        parts = ['records', name, raw_anchor]
        if pool:
            parts.append(pool)
        return _ce(value[weight], '/'.join(map(str, parts))), _pointer(*parts, role, field, weight)

    def receipt(name, anchor):
        r = receipt_map[name, anchor]
        return {'summary_sha256': r['summary_sha256'], 'receipt_sha256': r['receipt_sha256']}

    def mean(values):
        return math.fsum(values) / len(ANCHORS)

    tradeoffs, decomposition, missing_routes = [], [], []
    for name in complete:
        spec = specs[name]
        for label, weight in WEIGHTS.items():
            for role in ROLES:
                evidence = []
                for anchor in ANCHORS:
                    task_ce, task_pointer = get(name, anchor, TASK, selected, weight)
                    sensitive_ce, attack_pointer = get(name, anchor, 'attack:' + role, selected, weight)
                    h_ce, h_pointer = get('H', anchor, 'attack:' + role, selected, weight)
                    evidence.append({'anchor': anchor, **receipt(name, anchor),
                                     'H_summary_sha256': receipt('H', anchor)['summary_sha256'],
                                     'H_receipt_sha256': receipt('H', anchor)['receipt_sha256'],
                                     'task_pointer': task_pointer, 'attack_pointer': attack_pointer,
                                     'H_pointer': h_pointer, 'task_ce': task_ce,
                                     'sensitive_ce': sensitive_ce, 'H_sensitive_ce': h_ce,
                                     'sensitive_recovery': h_ce - sensitive_ce})
                tradeoffs.append({**spec, 'point_id': _digest([phase, name, role, label])[:20],
                                  'role': role, 'weighting': label,
                                  **{k: mean([e[k] for e in evidence]) for k in
                                     ('task_ce', 'sensitive_ce', 'H_sensitive_ce', 'sensitive_recovery')},
                                  'evidence': evidence})
            if spec['stage'] is None:
                continue
            for predictor, field in (('fixed', fixed), ('independent', independent)):
                evidence = []
                missing = []
                for anchor in ANCHORS:
                    metric = get(name, anchor, TASK, field, weight)
                    if metric is None:
                        missing.append(anchor)
                        continue
                    ce, pointer = metric
                    evidence.append({'anchor': anchor, **receipt(name, anchor),
                                     'metric_pointer': pointer, 'ce': ce})
                if missing:
                    missing_routes.append({'configuration': name, 'predictor': predictor,
                                           'weighting': label, 'missing_anchors': missing,
                                           'reason': 'Native summary has no complete decoder route'})
                else:
                    decomposition.append({**spec, 'point_id': _digest([phase, name, predictor, label])[:20],
                                          'weighting': label, 'predictor': predictor,
                                          'ce': mean([e['ce'] for e in evidence]), 'evidence': evidence})
    return {'schema': 1, 'phase': phase, 'pool': pool, 'scope': payload.get('scope'),
            'anchors': list(ANCHORS), 'weightings': list(WEIGHTS),
            'aggregation': 'Equal mean of three anchor estimates; no partial-anchor averages',
            'recovery_definition': 'H-only attacker CE minus release attacker CE; measured log-loss reduction, not mutual information',
            'utility_definition': 'Selected residence decoder CE, including permitted ancestors',
            'decomposition_definition': 'Fixed and independently selected residence decoder CE along the registered teacher/code/action/Q path; descriptive, not causal',
            'inference_performed': False,
            'configuration_metadata': specs, 'complete_configurations': complete,
            'unavailable_configurations': unavailable,
            'tradeoff_points': tradeoffs, 'decomposition_points': decomposition,
            'unavailable_decomposition': missing_routes}


def _tradeoff_style(row):
    category = row['category']
    if category == 'service_only':
        return dict(marker='X', color='#111111', s=64, zorder=5)
    if row['configuration'] == 'J':
        return dict(marker='*', color='#111111', s=100, zorder=5)
    if category == 'historical':
        return dict(marker='D', facecolors='none', edgecolors='#666666', s=38, zorder=3)
    if category == 'label_matched':
        color, marker = '#009E73', 's'
    else:
        color = {'T0': '#0072B2', 'Ttask': '#D55E00', 'Trisk': '#CC79A7'}[row['input']]
        marker = '^' if row['policy'] == 'L' else 'o'
    fine = row['conditioning_family'] != 'primary'
    return dict(marker=marker, facecolors='none' if row['actions'] == 33 else color,
                edgecolors='#111111' if fine else color, linewidths=1.15 if fine else .8,
                s=39, alpha=.85, zorder=4)


def _render(table, out):
    # Keep font/cache writes in the explicitly supplied figure directory.
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.grid': True, 'grid.alpha': .2, 'axes.axisbelow': True,
             'savefig.facecolor': 'white'}
    with matplotlib.rc_context(style):
        fig = Figure(figsize=(12, 13))
        FigureCanvasAgg(fig)
        axs = fig.subplots(4, 2, sharex=True, sharey='row')
        for i, role in enumerate(ROLES):
            for j, weight in enumerate(WEIGHTS):
                ax = axs[i, j]
                rows = [r for r in table['tradeoff_points'] if r['role'] == role and r['weighting'] == weight]
                ax.axhline(0, color='#444444', linewidth=.8, linestyle='--')
                for r in rows:
                    artist = ax.scatter(r['task_ce'], r['sensitive_recovery'], **_tradeoff_style(r))
                    artist.set_gid('point-' + r['point_id'])
                    if r['configuration'] in ('H', 'J'):
                        ax.annotate(r['configuration'], (r['task_ce'], r['sensitive_recovery']),
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_title(f'{role}  |  {weight}', loc='left', fontweight='bold')
                ax.margins(x=.13, y=.16)
                if j == 0:
                    ax.set_ylabel('Sensitive recovery relative to H\nH attacker CE − release attacker CE')
                if i == 3:
                    ax.set_xlabel('Residence task CE (selected decoder; lower is better)')
        handles = [Line2D([], [], linestyle='', marker=mk, color=col, markerfacecolor=face,
                          label=label, markersize=7) for mk, col, face, label in
                   [('X', '#111111', '#111111', 'H: service only'),
                    ('*', '#111111', '#111111', 'J: historical reference'),
                    ('D', '#666666', 'none', 'Historical releases'),
                    ('s', '#009E73', '#009E73', 'Label-matched controls / unconstrained'),
                    ('o', '#0072B2', '#0072B2', 'New Q: T0'),
                    ('o', '#D55E00', '#D55E00', 'New Q: Ttask'),
                    ('o', '#CC79A7', '#CC79A7', 'New Q: Trisk'),
                    ('^', '#555555', '#555555', 'L privacy policy'),
                    ('o', '#555555', '#555555', 'C privacy policy'),
                    ('o', '#555555', 'none', 'Expanded 33-action alphabet')]]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .962), ncol=3,
                   frameon=False, fontsize=8)
        fig.suptitle(f'Residence utility and sensitive recovery — {table["phase"]}', y=.995, fontsize=15)
        fig.text(.5, .012, 'All complete supplied configurations; equal three-anchor means. Positive recovery means lower attacker log loss than H.\n'
                 'Measured recovery is not mutual information. Descriptive full grid; no inference tests. Black Q outlines denote finer conditioning.',
                 ha='center', va='bottom', fontsize=8)
        fig.subplots_adjust(left=.12, right=.98, bottom=.085, top=.87, hspace=.32, wspace=.15)
        for extension in ('png', 'svg'):
            fig.savefig(out / f'utility_recovery.{extension}', dpi=180)

        fig2 = Figure(figsize=(13, 10))
        FigureCanvasAgg(fig2)
        axs = fig2.subplots(3, 2, sharex=True, sharey='row')
        labels = ('Continuous\nscore p', 'Input\ncode', 'Unprotected\naction', 'Protected\nL policy', 'Protected\nC policy')
        for i, code in enumerate(config.INPUTS):
            for j, weight in enumerate(WEIGHTS):
                ax = axs[i, j]
                rows = [r for r in table['decomposition_points'] if r['weighting'] == weight and
                        (r['input'] == code or r['stage'] == 'continuous')]
                for stage_index, stage in enumerate(STAGES):
                    names = sorted({r['configuration'] for r in rows if r['stage'] == stage})
                    offsets = {name: (0 if len(names) == 1 else -.22 + .44 * k / (len(names) - 1))
                               for k, name in enumerate(names)}
                    for r in (r for r in rows if r['stage'] == stage):
                        fixed = r['predictor'] == 'fixed'
                        color, marker = ('#0072B2', 'o') if fixed else ('#D55E00', '^')
                        artist = ax.scatter(stage_index + offsets[r['configuration']] + (-.045 if fixed else .045), r['ce'],
                                            marker=marker, s=32, alpha=.85,
                                            facecolors='none' if r['actions'] == 33 else color,
                                            edgecolors='#111111' if r['conditioning_family'] != 'primary' else color,
                                            linewidths=.8)
                        artist.set_gid('point-' + r['point_id'] + '-panel-' + code)
                ax.set_title(f'{code}  |  {weight}', loc='left', fontweight='bold')
                ax.set_xlim(-.5, 4.5)
                ax.set_xticks(range(5), labels)
                ax.tick_params(axis='x', labelbottom=True)
                if j == 0:
                    ax.set_ylabel('Residence CE (lower is better)')
                ax.margins(y=.15)
        fig2.legend(handles=[Line2D([], [], linestyle='', marker=m, color=c, label=l)
                             for m, c, l in [('o', '#0072B2', 'Fixed decoder'),
                                             ('^', '#D55E00', 'Independent learned decoder')]],
                    loc='upper center', bbox_to_anchor=(.5, .963), ncol=2, frameon=False)
        fig2.suptitle(f'Task path: continuous score → code / action → protected release — {table["phase"]}',
                      y=.995, fontsize=14)
        fig2.text(.5, .013, 'All available registered path settings; fixed lexical horizontal offsets separate configurations. Continuous score repeats across code panels.\n'
                  'No fixed decoder is imputed for a raw code. Hollow markers: 33 actions; black outlines: finer conditioning.\n'
                  'Equal three-anchor means; descriptive comparison, not a causal decomposition or a full-grid inference test.',
                  ha='center', va='bottom', fontsize=8)
        fig2.subplots_adjust(left=.085, right=.98, bottom=.11, top=.88, hspace=.4, wspace=.15)
        for extension in ('png', 'svg'):
            fig2.savefig(out / f'task_path_decomposition.{extension}', dpi=180)


def render_figures(grid_json, out_dir, metadata=None):
    """Write PNG/SVG figures, POINTS.json and a hash manifest to a fresh directory.

    ``grid_json`` is an explicit JSON path or an in-memory aggregate dictionary.
    Metadata maps configuration names to primary or registered branch specs.
    Existing nonempty directories are refused; input receipt hashes are
    recorded, not resolved by opening any private source artifacts.
    """
    if isinstance(grid_json, dict):
        payload = grid_json
        input_hash, input_kind = _digest(payload), 'canonical JSON object'
    else:
        source = Path(grid_json)
        payload = json.loads(source.read_text())
        input_hash, input_kind = _file_digest(source), 'explicit JSON file'
    table = build_point_table(payload, metadata=metadata)
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing to replace existing figure artifacts: {out}')
    out.mkdir(parents=True, exist_ok=True)
    table['input_sha256'] = input_hash
    table['metadata_sha256'] = _digest(metadata or {})
    (out / 'POINTS.json').write_text(json.dumps(table, indent=2, allow_nan=False) + '\n')
    _render(table, out)
    files = ('POINTS.json', 'utility_recovery.png', 'utility_recovery.svg',
             'task_path_decomposition.png', 'task_path_decomposition.svg')
    manifest = {'schema': 1, 'phase': table['phase'], 'pool': table['pool'],
                'input_kind': input_kind, 'input_sha256': input_hash,
                'metadata_sha256': table['metadata_sha256'],
                'figure_source_sha256': _file_digest(__file__),
                'complete_configurations': len(table['complete_configurations']),
                'unavailable_configurations': table['unavailable_configurations'],
                'tradeoff_points': len(table['tradeoff_points']),
                'decomposition_points': len(table['decomposition_points']),
                'inference_performed': False,
                'provenance_scope': 'Accepted-grid receipt declarations retained; private source files are not opened or independently authenticated here',
                'artifacts': {f: _file_digest(out / f) for f in files}}
    (out / 'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest
