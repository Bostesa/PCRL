"""Independent read-only paired-model, Adam, loss and gradient replay."""
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
from experiments.acs_bottleneck_training import tree_digest

SOURCES = {'A': ('income_binary', 'civilian_at_work'), 'B': ('public_coverage',)}
ROLES = {'A': ('public_coverage', 'SEX', 'RAC1P'), 'B': ('income_binary', 'civilian_at_work', 'SEX', 'RAC1P'), 'AB': ('SEX', 'RAC1P')}
TARGET_ORDER = ('SEX', 'RAC1P', 'income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')
COEFF = {'I': {'individual': -.1, 'extra_local': 0., 'coalition': 0.},
         'Iplus': {'individual': -.1, 'extra_local': -.1, 'coalition': 0.},
         'J': {'individual': -.1, 'extra_local': 0., 'coalition': -.1}}


def linear(x, state, prefix):
    return F.linear(x, state[prefix + '.weight'], state[prefix + '.bias'])


def literal_parts(state, x):
    h, logits = {}, {}
    for view, tasks in SOURCES.items():
        prefix = 'branches.' + view
        h[view] = linear(torch.relu(linear(x, state, prefix + '.mapper.0')), state, prefix + '.mapper.2')
        logits[view] = {task: linear(h[view], state, prefix + '.heads.' + task) for task in tasks}
    return h, logits


def initial_observers(interface, seed):
    dimensions = {'A': 16, 'B': 16, 'AB': 32} if interface == 'F' else {'A': 2, 'B': 1, 'AB': 3}
    states = {}
    for v, tasks in ROLES.items():
        for task in tasks:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260911 + 100*seed + 10*{'A': 0, 'B': 1, 'AB': 2}[v] + TARGET_ORDER.index(task))
                net = torch.nn.Sequential(torch.nn.Linear(dimensions[v], 64), torch.nn.ReLU(),
                    torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 9 if task == 'RAC1P' else 2))
            states.update({v+'__'+task+'.'+k: value for k, value in net.state_dict().items()})
    return states


def literal_diagnostic(state, observers, standardized, labels, priors, interface, condition):
    state = {name: value.detach().clone().requires_grad_(name.startswith('branches.')) for name, value in state.items()}
    x = torch.from_numpy(standardized)
    features, logits = literal_parts(state, x)
    tasks, source_purposes = {}, {}
    for v, names in SOURCES.items():
        for name in names:
            scores = logits[v][name].reshape(-1); y = torch.from_numpy(labels[name]); valid = y >= 0
            tasks[name] = F.binary_cross_entropy_with_logits(scores[valid], y[valid].float()) if valid.any() else scores.sum()*0.
        source_purposes[v] = torch.stack([tasks[name] for name in names]).mean()
    source = torch.stack(list(source_purposes.values())).mean()
    wires = dict(features) if interface == 'F' else {v: torch.cat([torch.sigmoid(logits[v][name]) for name in names], 1) for v, names in SOURCES.items()}
    wires['AB'] = torch.cat((wires['A'], wires['B']), 1)
    ce = {}
    for v, targets in ROLES.items():
        for target in targets:
            role = v+'__'+target; scores = wires[v]
            for layer in ('0', '2', '4'):
                scores = linear(scores, observers, role+'.'+layer)
                if layer != '4': scores = torch.relu(scores)
            y = torch.from_numpy(labels[target]); valid = y >= 0
            ce[role] = F.cross_entropy(scores[valid], y[valid]) if valid.any() else scores.sum()*0.
    local = {v: torch.stack([ce[v+'__'+target]/priors[target]['entropy'] for target in targets]).mean() for v, targets in ROLES.items()}
    extra = {v: torch.stack([ce[v+'__'+target]/priors[target]['entropy'] for target in ('SEX', 'RAC1P')]).mean() for v in ('A', 'B')}
    losses = {'source': source, 'individual': (local['A']+local['B'])/2., 'extra_local': extra['A']+extra['B'], 'coalition': local['AB']}
    coeff = {'source': 1., **(COEFF[condition] if condition else {'individual': 0., 'extra_local': 0., 'coalition': 0.})}
    groups = {v+'_mapper': [state['branches.'+v+'.mapper.'+k] for k in ('0.weight', '0.bias', '2.weight', '2.bias')] for v in ('A', 'B')}
    groups.update({v+'_heads': [state['branches.'+v+'.heads.'+task+'.'+k] for task in names for k in ('weight', 'bias')] for v, names in SOURCES.items()})
    groups['all_mappers'] = groups['A_mapper'] + groups['B_mapper']
    result = {'losses': {name: float(v.detach()) for name, v in losses.items()},
              'source_task_losses': {name: float(v.detach()) for name, v in tasks.items()},
              'observer_ce': {name: float(v.detach()) for name, v in ce.items()}, 'groups': {}, 'routing': {}}
    for group, params in groups.items():
        vectors = {}
        for name, value in losses.items():
            grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
        applied = {name: coeff[name]*value for name, value in vectors.items()}; record = {}
        for convention, values in (('raw', vectors), ('applied', applied)):
            for name, value in values.items(): record[name+'_'+convention+'_l2'] = float(torch.linalg.vector_norm(value))
            for a, b in (('source', 'individual'), ('source', 'coalition'), ('source', 'extra_local'), ('individual', 'extra_local'), ('extra_local', 'coalition'), ('individual', 'coalition')):
                dot = float(torch.dot(values[a], values[b])); denominator = record[a+'_'+convention+'_l2']*record[b+'_'+convention+'_l2']
                record[a+'_'+b+'_'+convention+'_dot'] = dot
                record[a+'_'+b+'_'+convention+'_cosine'] = dot/denominator if denominator else None
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
    return result


def compare(actual, expected):
    maximum = 0.
    for key, value in actual.items():
        if isinstance(value, dict): maximum = max(maximum, compare(value, expected[key]))
        elif value is None: assert expected[key] is None
        else:
            difference = abs(value-expected[key]); maximum = max(maximum, difference)
            assert np.isclose(value, expected[key], atol=1e-9, rtol=2e-6), (key, value, expected[key])
    return maximum


def verify_seed(out, cfg, seed, raw):
    directory = out/f'seed_{seed}'; training = directory/'training'; meta = check.read(training/'training.json')
    load = lambda name: torch.load(training/name, map_location='cpu', weights_only=True)
    olddir = ROOT/cfg['init_reference_results']/f'seed_{seed}'
    oldmeta = check.read(olddir/'training/training.json')
    genuine = torch.load(olddir/'training/initialization.pt', map_location='cpu', weights_only=True)['model_state']
    with np.load(directory/'split_rows.npz') as rows, np.load(olddir/'split_rows.npz') as oldrows:
        assert set(rows.files) == set(oldrows.files)
        for name in rows.files: np.testing.assert_array_equal(rows[name], oldrows[name])
        labels = {name: np.where(valid, y, -1).astype(np.int64) for name in TARGET_ORDER[:5]
                  for y, valid in [check.labels(raw.iloc[rows['representation_fit']], name)]}
        validation = {name: np.where(valid, y, -1).astype(np.int64) for name in TARGET_ORDER[2:5]
                      for y, valid in [check.labels(raw.iloc[rows['source_validation']], name)]}
    with np.load(ROOT/cfg['reference_results']/f'seed_{seed}/release_E_pca.npz') as saved:
        pca = saved['representation_fit'].copy(); pca_val = saved['source_validation'].copy()
    assert meta['source_label_hashes'] == oldmeta['source_label_hashes'] == {k: check.array_hash(labels[k]) for k in TARGET_ORDER[2:5]}
    assert meta['attribute_label_hashes'] == oldmeta['attribute_label_hashes'] == {k: check.array_hash(labels[k]) for k in TARGET_ORDER[:2]}
    assert meta['source_validation_label_hashes'] == oldmeta['source_validation_label_hashes'] == {k: check.array_hash(v) for k, v in validation.items()}
    assert check.array_hash(pca) == meta['fit_input_sha256'] and check.array_hash(pca_val) == meta['source_validation_input_sha256']
    for name, y in labels.items():
        counts = np.bincount(y[y >= 0], minlength=9 if name == 'RAC1P' else 2); p = counts/counts.sum()
        assert meta['prior_entropies'][name] == {'support': counts.tolist(), 'probabilities': p.tolist(), 'entropy': float(-(p[p>0]*np.log(p[p>0])).sum())}
    initial, warm = load('initialization.pt'), load('warm_base.pt'); state = initial['model_state']
    assert check.state_hash(state) == meta['initial_model_sha256'] == check.read(out/'PREFIT_IDENTITY.json')['seeds'][str(seed)]['initial_model_sha256']
    for name in ('input_mean', 'input_scale'): assert torch.equal(state[name], genuine[name])
    for v, tasks in SOURCES.items():
        for name, value in genuine.items():
            if name.startswith('mapper.') or any(name.startswith('heads.'+t+'.') for t in tasks): assert torch.equal(state['branches.'+v+'.'+name], value)
    assert sum(value.numel() for name, value in state.items() if name.startswith('branches.')) == 6355
    assert not any('decoder' in key for key in state)
    assert not initial['mapper_optimizer']['state'] and initial['adversary_optimizer'] is None and not initial['adversary_state']
    standardized = ((pca.astype(np.float64)-state['input_mean'].numpy())/state['input_scale'].numpy()).astype(np.float32)
    assert check.array_hash(standardized) == meta['standardized_fit_sha256']
    n = len(pca); batches = math_ceil(n/256)
    for phase, epochs, offset in (('warm_base', 60, 0), ('warm_adversary', 20, 100), ('continuation', 80, 200)):
        recorded = meta['schedules'][phase]; assert recorded == oldmeta['schedules'][phase]
        assert recorded['seed'] == 1280000+100*seed+offset
        rng = np.random.default_rng(recorded['seed']); digest = hashlib.sha256()
        for _ in range(epochs): digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest() == recorded['sha256']
    idx = np.random.default_rng(1280000+100*seed).permutation(n)[:256]
    assert check.array_hash(idx) == meta['diagnostic_indices_sha256']
    assert warm['counts'] == {'mapper_optimizer_steps': 60*batches, 'adversary_optimizer_steps': 0}
    points = []
    def count_adam(checkpoint, mapper, observer):
        assert len(checkpoint['mapper_optimizer']['state']) == 14
        assert {int(value['step']) for value in checkpoint['mapper_optimizer']['state'].values()} == {mapper}
        if observer:
            assert len(checkpoint['adversary_optimizer']['state']) == 54
            assert {int(value['step']) for value in checkpoint['adversary_optimizer']['state'].values()} == {observer}
    count_adam(warm, 60*batches, 0)
    for interface in ('F', 'P'):
        original_obs = initial_observers(interface, seed)
        observer_initial = load(interface+'/observer_initialization.pt'); common = load(interface+'/warm_adversary.pt')
        assert check.state_hash(observer_initial['adversary_state']) == check.state_hash(original_obs)
        for ck in (observer_initial, common):
            assert tree_digest(ck['model_state']) == tree_digest(warm['model_state'])
            assert tree_digest(ck['mapper_optimizer']) == tree_digest(warm['mapper_optimizer'])
        count_adam(common, 60*batches, 20*batches)
        points.extend([(initial['model_state'], original_obs, interface, None, meta['diagnostics']['I'][interface]),
                       (warm['model_state'], original_obs, interface, None, meta['diagnostics']['W'][interface]),
                       (common['model_state'], common['adversary_state'], interface, 'J', meta['diagnostics'][interface+'_fork'])])
        for regime in ('I', 'Iplus', 'J'):
            name = interface+'_'+regime; fork, final = load(name+'/fork.pt'), load(name+'/final.pt'); record = meta['arms'][name]
            for key in ('model_state', 'adversary_state', 'mapper_optimizer', 'adversary_optimizer', 'counts'):
                assert tree_digest(fork[key]) == tree_digest(common[key])
            assert torch.equal(fork['torch_rng_state'], common['torch_rng_state'])
            assert fork['schedule_state']['schedules'] == common['schedule_state']['schedules'] == meta['schedules']
            assert fork['schedule_state']['phase'] == 'continuation' and fork['schedule_state']['completed_epochs'] == 0
            assert final['schedule_state']['phase'] == 'continuation' and final['schedule_state']['completed_epochs'] == 80
            assert final['schedule_state']['next_minibatch_index'] == 0
            assert final['counts'] == {'mapper_optimizer_steps': 140*batches, 'adversary_optimizer_steps': 260*batches}
            count_adam(final, 140*batches, 260*batches)
            assert record['coefficients'] == COEFF[regime] and record['schedule_sha256'] == meta['schedules']['continuation']['sha256']
            assert record['mapper_exposure_per_row'] == 140 and record['observer_exposure_per_row'] == 260
            for role, value in record['observer_valid_label_exposures'].items(): assert value == int((labels[role.split('__')[1]]>=0).sum())*260
            for task, value in record['source_valid_label_exposures'].items(): assert value == int((labels[task]>=0).sum())*140
            assert check.state_hash(final['model_state']) == record['final_hashes']['model']
            assert check.state_hash(final['adversary_state']) == record['final_hashes']['adversaries']
            assert tree_digest(final['mapper_optimizer']) == record['final_hashes']['mapper_optimizer']
            assert tree_digest(final['adversary_optimizer']) == record['final_hashes']['adversary_optimizer']
            points.extend([(fork['model_state'], fork['adversary_state'], interface, regime, record['fork_diagnostic']),
                           (final['model_state'], final['adversary_state'], interface, regime, record['final_diagnostic'])])
    maximum = 0.
    for state, observers, interface, regime, recorded in points:
        actual = literal_diagnostic(state, observers, standardized[idx], {name: y[idx] for name, y in labels.items()}, meta['prior_entropies'], interface, regime)
        maximum = max(maximum, compare(actual, recorded))
        assert recorded['state_unchanged'] and recorded['rng_unchanged']
    return {'seed': seed, 'passed': True, 'gradient_points_replayed': len(points), 'maximum_gradient_scalar_error': maximum,
            'exact_initial_heads_mappers': True, 'exact_forks_full_Adam': True, 'matching_schedules_exposures': True,
            'training_json_sha256': check.sha(training/'training.json')}


def math_ceil(value):
    return int(np.ceil(value))


def verify_execution_chain(out, cfg, frozen):
    """Preserve original hashes and accept only an explicit checked chain."""
    expected = dict(frozen['sha256'])
    amendment_path = out/'EXECUTION_AMENDMENTS.json'
    record = {'amendments_sha256': None, 'amendments': [], 'preserved_artifacts_checked': 0}
    if amendment_path.exists():
        amendments = check.read(amendment_path)
        assert amendments['historical_metadata_anchor_commit'] == cfg['starting_commit']
        for path, digest in amendments.get('historical_metadata_anchor_hashes', {}).items():
            assert check.sha(ROOT/path) == digest
        for amendment in amendments['amendments']:
            assert amendment['numerical_recipe_changed'] is False
            assert amendment['scientific_results_invalidated'] is False
            assert amendment['new_optimization_in_recovery_preflight'] == 0
            # This operational recovery changes loader/orchestration only.
            assert set(amendment['source_changes']) == {'experiments/run_acs_coalition.py'}
            for path, change in amendment['source_changes'].items():
                assert expected[path] == change['before_sha256']
                assert check.sha(ROOT/change['preserved_source_path']) == change['before_sha256']
                expected[path] = change['after_sha256']
            preserved_path = out/amendment['preserved_artifact_record']
            preserved = check.read(preserved_path)
            for path, digest in preserved['files_sha256'].items():
                assert check.sha(ROOT/path) == digest, ('Pre-recovery artifact changed', path)
            record['preserved_artifacts_checked'] += len(preserved['files_sha256'])
            record['amendments'].append({'id': amendment['id'], 'preserved_artifact_record_sha256': check.sha(preserved_path),
                'completed_pairs_preserved': amendment['completed_pairs_preserved'],
                'completed_audit_conditions_preserved': amendment['completed_audit_conditions_preserved'],
                'numerical_recipe_changed': False, 'scientific_results_invalidated': False})
        record['amendments_sha256'] = check.sha(amendment_path)
    for path, digest in expected.items(): assert check.sha(ROOT/path) == digest
    for group in ('reference_record_hashes', 'required_local_reference_hashes'):
        for path, digest in frozen.get(group, {}).items(): assert check.sha(ROOT/path) == digest
    assert check.sha(out/'PREFIT_IDENTITY.json') == frozen['prefit_identity_sha256']
    record['original_protocol_freeze_sha256'] = check.sha(out/'protocol_freeze.json')
    return record


def verify(out):
    started = time.perf_counter(); cfg = check.read(out/'config.json'); frozen = check.read(out/'protocol_freeze.json')
    execution_chain = verify_execution_chain(out, cfg, frozen)
    complete = [seed for seed in (0, 1, 2) if (out/f'seed_{seed}/complete.json').exists()]
    assert complete
    parent = ROOT/cfg['parent_results']; rawpath = ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(rawpath) == check.read(parent/'schema_support.json')['raw_sha256']
    raw = pd.read_csv(rawpath, usecols=['PINCP', 'ESR', 'PUBCOV', 'SEX', 'RAC1P'])
    reports = [verify_seed(out, cfg, seed, raw) for seed in complete]
    return {'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Independent literal tensor gradients and saved initial/fork/Adam/data/exposure checks; no fitting; candidate inference handled separately',
            'completed_seeds': complete, 'expected_seeds': [0, 1, 2], 'complete_matrix': complete == [0, 1, 2],
            'gradient_points_replayed': sum(row['gradient_points_replayed'] for row in reports),
            'maximum_gradient_scalar_error': max(row['maximum_gradient_scalar_error'] for row in reports),
            'seeds': reports, 'execution_chain': execution_chain,
            'script_sha256': check.sha(__file__), 'runtime_seconds': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/redesign_20260908_acs_coalition_v1')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError('Never overwrite completed replay evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle: handle.write(rendered)
    else: print(rendered, end='')


if __name__ == '__main__': main()
