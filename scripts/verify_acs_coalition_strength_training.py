"""Read-only independent training replay for new strength continuations.

Historical forks/metadata are hash-bound, never refitted. Literal tensor
operations reconstruct new gradients; auditor/release scoring is separate.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_coalition_training import literal_parts, linear, SOURCES, ROLES, TARGET_ORDER, compare
from experiments.acs_bottleneck_training import tree_digest

BETAS = (0., .025, .05, .1, .2)


def coefficients(regime, beta):
    assert regime in ('Iplus', 'J') and beta in BETAS
    return {'source': 1., 'individual': -.1, 'extra_local': -beta if regime == 'Iplus' else 0.,
            'coalition': -beta if regime == 'J' else 0.}


def literal_components(original, observers, standardized, labels, priors, interface):
    state = {name: value.detach().clone().requires_grad_(name.startswith('branches.')) for name, value in original.items()}
    features, logits = literal_parts(state, torch.from_numpy(standardized))
    tasks, purposes = {}, {}
    for view, names in SOURCES.items():
        for name in names:
            scores = logits[view][name].reshape(-1); y = torch.from_numpy(labels[name]); valid = y >= 0
            tasks[name] = F.binary_cross_entropy_with_logits(scores[valid], y[valid].float()) if valid.any() else scores.sum()*0.
        purposes[view] = torch.stack([tasks[name] for name in names]).mean()
    source = torch.stack(list(purposes.values())).mean()
    wires = dict(features) if interface == 'F' else {view: torch.cat([torch.sigmoid(logits[view][name]) for name in names], 1) for view, names in SOURCES.items()}
    wires['AB'] = torch.cat((wires['A'], wires['B']), 1)
    ce = {}
    for view, targets in ROLES.items():
        for target in targets:
            role = view+'__'+target; scores = wires[view]
            for layer in ('0', '2', '4'):
                scores = linear(scores, observers, role+'.'+layer)
                if layer != '4': scores = torch.relu(scores)
            y = torch.from_numpy(labels[target]); valid = y >= 0
            ce[role] = F.cross_entropy(scores[valid], y[valid]) if valid.any() else scores.sum()*0.
    local = {view: torch.stack([ce[view+'__'+target]/priors[target]['entropy'] for target in targets]).mean() for view, targets in ROLES.items()}
    extra = {view: torch.stack([ce[view+'__'+target]/priors[target]['entropy'] for target in ('SEX', 'RAC1P')]).mean() for view in ('A', 'B')}
    losses = {'source': source, 'individual': (local['A']+local['B'])/2., 'extra_local': extra['A']+extra['B'], 'coalition': local['AB']}
    return state, losses, tasks, ce, local, extra


def vector(value, parameters):
    grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, grads)])


def diagnostic(original, observers, standardized, labels, priors, interface, regime, beta, *, verify_affine=False):
    state, losses, tasks, ce, local, extra = literal_components(original, observers, standardized, labels, priors, interface)
    coeff = coefficients(regime, beta)
    groups = {view+'_mapper': [state['branches.'+view+'.mapper.'+key] for key in ('0.weight', '0.bias', '2.weight', '2.bias')] for view in ('A', 'B')}
    groups.update({view+'_heads': [state['branches.'+view+'.heads.'+task+'.'+key] for task in names for key in ('weight', 'bias')] for view, names in SOURCES.items()})
    groups['all_mappers'] = groups['A_mapper']+groups['B_mapper']
    result = {'losses': {name: float(value.detach()) for name, value in losses.items()},
              'source_task_losses': {name: float(value.detach()) for name, value in tasks.items()},
              'observer_ce': {name: float(value.detach()) for name, value in ce.items()}, 'groups': {}, 'routing': {}}
    for group, params in groups.items():
        raw = {name: vector(value, params) for name, value in losses.items()}
        applied = {name: coeff[name]*value for name, value in raw.items()}; record = {}
        for convention, values in (('raw', raw), ('applied', applied)):
            for name, value in values.items(): record[name+'_'+convention+'_l2'] = float(torch.linalg.vector_norm(value))
            for left, right in (('source', 'individual'), ('source', 'coalition'), ('source', 'extra_local'), ('individual', 'extra_local'), ('extra_local', 'coalition'), ('individual', 'coalition')):
                dot = float(torch.dot(values[left], values[right])); denominator = record[left+'_'+convention+'_l2']*record[right+'_'+convention+'_l2']
                record[left+'_'+right+'_'+convention+'_dot'] = dot
                record[left+'_'+right+'_'+convention+'_cosine'] = dot/denominator if denominator else None
        record['combined_protection_l2'] = float(torch.linalg.vector_norm(applied['individual']+applied['extra_local']+applied['coalition']))
        record['objective_l2'] = float(torch.linalg.vector_norm(sum(applied.values())))
        result['groups'][group] = record
    for owner in ('A', 'B'):
        for dest in ('A', 'B'):
            for name, value in (('local', local[owner]), ('extra', extra[owner])):
                grads = torch.autograd.grad(value, groups[dest+'_mapper'], retain_graph=True, allow_unused=True)
                norm = float(torch.sqrt(sum((g*g).sum() for g in grads if g is not None))) if any(g is not None for g in grads) else 0.
                result['routing'][f'{name}_{owner}_to_{dest}_mapper_l2'] = norm
                if owner != dest: assert norm == 0.
    affine = None
    if verify_affine:
        params = [value for name, value in state.items() if name.startswith('branches.')]
        base_loss = losses['source']-.1*losses['individual']; base_gradient = vector(base_loss, params)
        maximum_gradient_error, maximum_loss_error = 0., 0.
        for family in ('Iplus', 'J'):
            term = losses['extra_local' if family == 'Iplus' else 'coalition']; signed_unit = vector(-term, params)
            for b in BETAS:
                objective = base_loss-b*term if b else base_loss
                actual = vector(objective, params)-base_gradient; expected = b*signed_unit
                maximum_gradient_error = max(maximum_gradient_error, float(torch.max(torch.abs(actual-expected))))
                maximum_loss_error = max(maximum_loss_error, abs(float((objective-base_loss+b*term).detach())))
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-5)
                torch.testing.assert_close(objective-base_loss, -b*term, atol=8e-8, rtol=3e-6)
                if b: assert not torch.allclose(vector(objective, params), expected)
        affine = {'all_five_strengths_both_regimes': True, 'full_gradient_affine_not_proportional': True,
                  'maximum_gradient_difference_error': maximum_gradient_error, 'maximum_loss_difference_error': maximum_loss_error,
                  'gradient_difference_tolerance': {'atol': 2e-7, 'rtol': 3e-5}}
    return result, affine


def seed_context(out, cfg, seed, raw, reuse):
    parent = ROOT/cfg['coalition_reference_results']/f'seed_{seed}'
    hist_path = parent/'training/training.json'
    assert check.sha(hist_path) == reuse['historical_files_sha256'][str(hist_path.relative_to(ROOT))]
    meta = check.read(hist_path)
    with np.load(out/f'seed_{seed}/split_rows.npz') as rows, np.load(parent/'split_rows.npz') as historical_rows:
        assert set(rows.files) == set(historical_rows.files)
        for name in rows.files: np.testing.assert_array_equal(rows[name], historical_rows[name])
        labels = {name: np.where(valid, y, -1).astype(np.int64) for name in TARGET_ORDER[:5]
                  for y, valid in [check.labels(raw.iloc[rows['representation_fit']], name)]}
        validation = {name: np.where(valid, y, -1).astype(np.int64) for name in TARGET_ORDER[2:5]
                      for y, valid in [check.labels(raw.iloc[rows['source_validation']], name)]}
    with np.load(ROOT/cfg['reference_results']/f'seed_{seed}/release_E_pca.npz') as saved:
        pca, pv = saved['representation_fit'].copy(), saved['source_validation'].copy()
    assert check.array_hash(pca) == meta['fit_input_sha256'] and check.array_hash(pv) == meta['source_validation_input_sha256']
    assert {name: check.array_hash(y) for name, y in labels.items() if name in TARGET_ORDER[2:5]} == meta['source_label_hashes']
    assert {name: check.array_hash(y) for name, y in labels.items() if name in TARGET_ORDER[:2]} == meta['attribute_label_hashes']
    assert {name: check.array_hash(y) for name, y in validation.items()} == meta['source_validation_label_hashes']
    priors = {}
    for name, y in labels.items():
        counts = np.bincount(y[y >= 0], minlength=9 if name=='RAC1P' else 2); p = counts/counts.sum()
        priors[name] = {'support': counts.tolist(), 'probabilities': p.tolist(), 'entropy': float(-(p[p>0]*np.log(p[p>0])).sum())}
    assert priors == meta['prior_entropies']
    mean, std = pca.astype(np.float64).mean(0), pca.astype(np.float64).std(0)
    scale = np.where(std > 1e-12, std, 1.)
    np.testing.assert_array_equal(mean, meta['preprocessing']['mean']); np.testing.assert_array_equal(scale, meta['preprocessing']['scale'])
    standardized = ((pca.astype(np.float64)-mean)/scale).astype(np.float32)
    idx = np.random.default_rng(1280000+100*seed).permutation(len(pca))[:256]
    assert check.array_hash(idx) == meta['diagnostic_indices_sha256']
    forks = {}
    for interface in ('F', 'P'):
        path = parent/'training'/(interface+'_I')/'fork.pt'
        assert check.sha(path) == reuse['historical_files_sha256'][str(path.relative_to(ROOT))]
        forks[interface] = torch.load(path, map_location='cpu', weights_only=True)
    return {'meta': meta, 'pca': pca, 'validation_pca': pv, 'labels': labels, 'validation': validation,
            'priors': priors, 'standardized': standardized, 'indices': idx, 'forks': forks}


def verify_unit(out, cfg, entry, context, affine_seen):
    directory = out/f"seed_{entry['seed']}"/'training'/entry['condition']
    complete = check.read(directory/'complete.json')
    for path, digest in complete['files_sha256'].items(): assert check.sha(ROOT/path) == digest
    meta = check.read(directory/'training.json'); source = context['meta']
    assert meta['study'] == meta['config']['study'] == 'acs_coalition_strength_v1' and not meta['miniature']
    assert meta['beta'] == entry['beta'] and meta['regime'] == entry['family'] and meta['interface'] == entry['interface'] and meta['seed'] == entry['seed']
    assert entry['beta'] in (.025, .05, .2)
    assert meta['coefficients'] == {k: v for k, v in coefficients(entry['family'], entry['beta']).items() if k != 'source'}
    assert not meta['reserved_labels_received'] and not meta['final_evaluation_received'] and not meta['warmup_refitted']
    assert meta['historical_fork_unchanged'] and meta['caller_rng_unchanged']
    assert meta['source_label_hashes'] == source['source_label_hashes'] and meta['attribute_label_hashes'] == source['attribute_label_hashes']
    assert meta['source_validation_label_hashes'] == source['source_validation_label_hashes']
    assert meta['fit_input_sha256'] == source['fit_input_sha256'] and meta['source_validation_input_sha256'] == source['source_validation_input_sha256']
    assert meta['preprocessing'] == source['preprocessing'] and meta['prior_entropies'] == context['priors']
    assert meta['historical_training_module_sha256'] == source['module_source_sha256']
    assert meta['observer_optimization'] == source['observer_optimization']
    assert meta['observer_roles'] == source['observer_roles'][entry['interface']]
    original = context['forks'][entry['interface']]
    fork = torch.load(directory/'fork.pt', map_location='cpu', weights_only=True)
    final = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
    assert tree_digest(fork) == tree_digest(original) == meta['historical_fork_tree_sha256']
    assert check.sha(ROOT/entry['historical_fork']) == meta['historical_fork_sha256']
    assert Path(meta['historical_fork_path']).resolve() == (ROOT/entry['historical_fork']).resolve()
    assert meta['fork_hashes'] == source['arms'][entry['interface']+'_'+entry['family']]['fork_hashes']
    n = len(context['pca']); batches = (n+255)//256
    assert fork['counts'] == {'mapper_optimizer_steps': 60*batches, 'adversary_optimizer_steps': 20*batches}
    assert final['counts'] == meta['counts'] == {'mapper_optimizer_steps': 140*batches, 'adversary_optimizer_steps': 260*batches}
    for checkpoint, mp, obs in ((fork, 60*batches, 20*batches), (final, 140*batches, 260*batches)):
        assert len(checkpoint['mapper_optimizer']['state']) == 14 and len(checkpoint['adversary_optimizer']['state']) == 54
        assert {int(v['step']) for v in checkpoint['mapper_optimizer']['state'].values()} == {mp}
        assert {int(v['step']) for v in checkpoint['adversary_optimizer']['state'].values()} == {obs}
        for name in ('mapper_optimizer', 'adversary_optimizer'):
            for group in checkpoint[name]['param_groups']:
                assert group['lr'] == .001 and group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0.
    assert final['schedule_state']['phase'] == 'continuation' and final['schedule_state']['completed_epochs'] == 80 and final['schedule_state']['next_minibatch_index'] == 0
    assert final['schedule_state']['schedules'] == fork['schedule_state']['schedules'] == meta['schedules'] == source['schedules']
    assert torch.equal(final['torch_rng_state'], fork['torch_rng_state'])
    for phase, epochs, offset in (('warm_base', 60, 0), ('warm_adversary', 20, 100), ('continuation', 80, 200)):
        seed = 1280000+100*entry['seed']+offset; schedule = meta['schedules'][phase]; assert schedule['seed'] == seed
        rng = np.random.default_rng(seed); digest = hashlib.sha256()
        for _ in range(epochs): digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest() == schedule['sha256']
    assert meta['schedule_sha256'] == meta['schedules']['continuation']['sha256']
    assert meta['selected_epoch'] == 80 and meta['mapper_exposure_per_row'] == 140 and meta['observer_exposure_per_row'] == 260
    for name, y in context['labels'].items():
        if name in TARGET_ORDER[2:5]: assert meta['source_valid_label_exposures'][name] == int((y>=0).sum())*140
    for role, value in meta['observer_valid_label_exposures'].items(): assert value == int((context['labels'][role.split('__')[1]]>=0).sum())*260
    assert check.state_hash(final['model_state']) == meta['final_hashes']['model']
    assert check.state_hash(final['adversary_state']) == meta['final_hashes']['adversaries']
    assert tree_digest(final['mapper_optimizer']) == meta['final_hashes']['mapper_optimizer']
    assert tree_digest(final['adversary_optimizer']) == meta['final_hashes']['adversary_optimizer']
    assert meta['standardized_fit_sha256'] == check.array_hash(context['standardized'])
    idx = context['indices']; assert meta['diagnostic_indices_sha256'] == check.array_hash(idx)
    maximum = 0.; affine = None
    for point, cp in (('fork_diagnostic', fork), ('final_diagnostic', final)):
        key = (entry['seed'], entry['interface'])
        check_affine = point == 'fork_diagnostic' and key not in affine_seen
        actual, a = diagnostic(cp['model_state'], cp['adversary_state'], context['standardized'][idx],
            {name: y[idx] for name, y in context['labels'].items()}, context['priors'], entry['interface'], entry['family'], entry['beta'], verify_affine=check_affine)
        maximum = max(maximum, compare(actual, meta[point]))
        assert meta[point]['coefficients'] == coefficients(entry['family'], entry['beta'])
        assert meta[point]['state_unchanged'] and meta[point]['rng_unchanged']
        if a is not None: affine = a; affine_seen.add(key)
    return {'seed': entry['seed'], 'condition': entry['condition'], 'passed': True, 'gradient_points_replayed': 2,
            'maximum_gradient_scalar_error': maximum, 'affine_at_fixed_historical_fork': affine,
            'exact_full_historical_fork': True, 'historical_prefix_refits': 0, 'all_Adam_schedule_RNG_exposure_checks': True,
            'training_json_sha256': check.sha(directory/'training.json')}


def verify(out):
    started = time.perf_counter(); cfg = check.read(out/'config.json'); freeze = check.read(out/'protocol_freeze.json')
    for path, digest in freeze['scientific_and_protocol_sha256'].items(): assert check.sha(ROOT/path) == digest
    assert check.sha(out/'REUSE_MANIFEST.json') == freeze['reuse_manifest_sha256']
    assert check.sha(out/'PREFIT_IDENTITY.json') == freeze['prefit_identity_sha256']
    reuse = check.read(out/'REUSE_MANIFEST.json')
    entries = [e for e in reuse['systems'] if not e['reused'] and (out/f"seed_{e['seed']}"/'training'/e['condition']/'complete.json').exists()]
    assert entries
    parent = ROOT/cfg['parent_results']; rawpath = ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(rawpath) == check.read(parent/'schema_support.json')['raw_sha256']
    raw = pd.read_csv(rawpath, usecols=['PINCP', 'ESR', 'PUBCOV', 'SEX', 'RAC1P'])
    contexts, reports, affine_seen = {}, [], set()
    for entry in entries:
        seed = entry['seed']
        if seed not in contexts: contexts[seed] = seed_context(out, cfg, seed, raw, reuse)
        reports.append(verify_unit(out, cfg, entry, contexts[seed], affine_seen))
    return {'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Independent new-continuation fullfork/Adam/RNG/data/exposure and literal gradient replay; historical forks bound without refitting or historical auditor inference',
            'new_units_checked': len(reports), 'expected_new_units': 36, 'complete_matrix': len(reports)==36,
            'gradient_points_replayed': sum(r['gradient_points_replayed'] for r in reports),
            'maximum_gradient_scalar_error': max(r['maximum_gradient_scalar_error'] for r in reports),
            'fixed_forks_affine_checked': len(affine_seen),
            'maximum_affine_gradient_difference_error': max(r['affine_at_fixed_historical_fork']['maximum_gradient_difference_error'] for r in reports if r['affine_at_fixed_historical_fork']),
            'units': reports, 'script_sha256': check.sha(__file__), 'runtime_seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/redesign_20260909_acs_coalition_strength_v1')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError('Preserve completed replay evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle: handle.write(rendered)
    else: print(rendered, end='')


if __name__=='__main__': main()
