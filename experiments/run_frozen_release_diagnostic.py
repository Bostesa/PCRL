"""Diagnose saved adversaries on immutable purpose releases; never train encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from experiments import run_nonlinear_conflict as old
from experiments.nonlinear_conflict_training import state_digest, _model_hash
from experiments.nonlinear_release_training import _architecture, ReleaseAdversaries
from experiments.nonlinear_conflict_probes import (
    FitPreprocessing, MLPProbe, AffineProbe, ProbeConfig, predictive_scores,
)
from pcrl.models.lora import PerPurposeLoRAEncoder

torch.set_num_threads(1)
PRIOR = ROOT / 'results/redesign_20260907_nonlinear_release_v1'
CONFIG = {
    'seeds': [2, 0, 1], 'test_rng_seeds': {str(s): 1000004+100*s for s in (0, 1, 2)},
    'test_n': 4096, 'epochs': 80, 'batch_size': 256, 'validation_interval': 5,
    'lr': .001, 'trajectories_per_B_C': 2,
    'bridge': 'Only seed2 E_dual_0.01: if any validation D-max(B,C) > .05, C with attacker-fit coordinate statistics; identical other settings.',
    'reference_D': 'Exact saved independent audit checkpoints; no refitting. 2 starts x 640 original steps per network.',
}
dump = old.dump


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_records(module):
    return {group: {name: {'sha256': old.digest_array(t.detach().numpy()),
                         'shape': list(t.shape), 'dtype': str(t.dtype)}
                    for name, t in values}
            for group, values in [('parameters', module.named_parameters()), ('buffers', module.named_buffers())]}


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def load_frozen(source):
    """Reconstruct saved tensors and maps, with no fitting or optimizer restore."""
    source = Path(source)
    c = torch.load(source/'training/final.pt', map_location='cpu', weights_only=True)
    cfg = json.loads((source/'training/metadata.json').read_text())['config']
    encoder = PerPurposeLoRAEncoder(_architecture(c['encoder'], 'backbone.'), 2,
        rank=cfg['rank'], alpha=cfg['alpha'], dropout=cfg['dropout'], lora_target='all_linear')
    encoder.load_state_dict(c['encoder'])
    width = c['heads']['0.weight'].shape[1]
    heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
    heads.load_state_dict(c['heads'])
    adversaries = ReleaseAdversaries(width, cfg['adversary_hidden'])
    adversaries.load_state_dict(c['adversaries'])
    for m in (encoder, heads, adversaries):
        freeze(m)
    if _model_hash(encoder, heads) != c['state_hash']:
        raise ValueError('Saved encoder/head content hash mismatch')
    with np.load(source/'erasers.npz') as z:
        erasers = {key: z[key].copy() for key in z.files}
    maps = {}
    for step in (375, 400):
        path = source/f'training/training_eraser_step_{step}.pt'
        if path.exists():
            maps[str(step)] = torch.load(path, map_location='cpu', weights_only=True)['release']
    if '400' in maps and state_digest(maps['400']) != state_digest(c['training_release']):
        raise ValueError('Saved final training-map checkpoint differs from final model buffer')
    return {'encoder': encoder, 'heads': heads, 'adversaries': adversaries,
            'checkpoint': c, 'erasers': erasers, 'training_maps': maps}


def integrity(model):
    return {'encoder': tensor_records(model['encoder']), 'heads': tensor_records(model['heads']),
            'saved_adversaries': tensor_records(model['adversaries']),
            'final_erasers': {k: old.digest_array(v) for k, v in model['erasers'].items()},
            'training_map_buffers': {k: state_digest(v) for k, v in model['training_maps'].items()},
            'saved_coordinates': state_digest(model['checkpoint']['training_release']),
            'target_mean': old.digest_array(model['checkpoint']['target_mean'].numpy()),
            'target_std': old.digest_array(model['checkpoint']['target_std'].numpy())}


def apply_final(raw, erasers):
    # Historical files preserve the exact dense P/center, not factorized LEACE.
    # Float64 affine reconstruction differs only in operation ordering.
    views = [h-(h-erasers[f'p{p+1}_center']) @ (np.eye(h.shape[1])-erasers[f'p{p+1}_matrix']).T
             for p, h in enumerate(raw)]
    return views + [np.concatenate(views, axis=1)]


def apply_training_map(raw, state):
    views = [(torch.from_numpy(h).float() @ state['projections'][p].T + state['biases'][p]).double().numpy()
             for p, h in enumerate(raw)]
    return views + [np.concatenate(views, axis=1)]


def cache_releases(model, data, directory):
    directory.mkdir(parents=True, exist_ok=True)
    views, raw, hashes = {}, {}, {}
    for split, row in data.items():
        raw[split] = [old.encode(model['encoder'], row['x'], p) for p in range(2)]
        views[split] = apply_final(raw[split], model['erasers'])
        arrays = {f'view_{p}': h for p, h in enumerate(views[split])}
        arrays.update(y=row['y'], ids=row['ids'])
        np.savez_compressed(directory/f'{split}.npz', **arrays)
        hashes[split] = {k: old.digest_array(v) for k, v in arrays.items()}
    return views, raw, hashes


def load_audit(source):
    result = {}
    for p, label in enumerate(('p1', 'p2', 'combined')):
        for kind in ('mlp', 'linear'):
            directory = Path(source)/'probes'/f'attack_{label}_{kind}'
            meta = json.loads((directory/'metadata.json').read_text())
            with np.load(directory/'preprocessing.npz') as z:
                pp = FitPreprocessing(**{k: z[k].copy() for k in z.files})
            names = meta['target_names']
            if kind == 'mlp':
                states = []
                for j, name in enumerate(names):
                    c = torch.load(directory/f'selected_target_{j}.pt', weights_only=True, map_location='cpu')
                    if c['target_index'] != j or c['target_name'] != name:
                        raise ValueError('Audit target identity mismatch')
                    states.append(c['state_dict'])
                probe = MLPProbe(pp, states, names, meta, ProbeConfig(**meta['config']))
            else:
                with np.load(directory/'model.npz') as z:
                    coefficients = z['coefficients'].copy()
                probe = AffineProbe(pp, coefficients, names, meta)
            result[(p, kind)] = probe
    return result


def audit_score(audits, views, y, kind='mlp'):
    from experiments.frozen_release_adversaries import TARGET_NAMES
    scores = {}
    for p in range(3):
        scores.update(audits[p, kind].score(views[p], y))
    return {k: scores[k] for k in TARGET_NAMES}


def diagnostic_test(seed, manifest, selection_path):
    if not Path(selection_path).exists():
        raise RuntimeError('Save attacker selection before diagnostic test generation')
    rng_seed = CONFIG['test_rng_seeds'].get(str(seed), 1000004+100*seed)
    latent = np.random.default_rng(rng_seed).normal(size=(CONFIG['test_n'], 8))
    q1, q2, offset = old.generator_parameters()
    r = latent @ q1
    x = ((r+.2*r**3)@q2+offset-np.array(manifest['preprocessing_mean']))/np.array(manifest['preprocessing_std'])
    return {'x': x, 'y': latent[:, :3], 'ids': np.arange(len(x), dtype=np.int64)+np.int64(rng_seed)*100000}


def selected_keys(prior_seed):
    selection = json.loads((Path(prior_seed)/'selection_before_test.json').read_text())['selection']
    keys = [selection[k]['key'] for k in ('E', 'D')]
    return keys, selection


def run_seed(seed, out, prior=PRIOR):
    from experiments.frozen_release_adversaries import SavedAdversaryPredictor, fit_continuations, TARGET_NAMES
    tick = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    prior_seed = Path(prior)/f'seed_{seed}'
    data, manifest = old.make_data(seed, include_test=False)
    if manifest != json.loads((prior_seed/'split_manifest_fit.json').read_text()):
        raise ValueError('Diagnostic fitting data differs from historical streams')
    dump(out/'split_manifest_fit.json', manifest)
    np.savez_compressed(out/'split_ids_fit.npz', **{k: v['ids'] for k, v in data.items()})
    # No fitting routine receives representation training or calibration arrays.
    fitdata = {k: data[k] for k in ('attacker_fit', 'validation')}
    keys, selections = selected_keys(prior_seed)
    if seed == 2 and 'E_dual_0.01' not in keys:
        keys.insert(0, 'E_dual_0.01')
    result = {'seed': seed, 'prior_selection': selections, 'methods': {}, 'controls': {}}
    prepared = {}
    for key in keys:
        print(f'seed={seed}: frozen diagnostic {key}', flush=True)
        directory = out/key
        model = load_frozen(prior_seed/key)
        before = integrity(model)
        views, raw, hashes = cache_releases(model, fitdata, directory/'release_cache')
        np.savez_compressed(directory/'frozen_final_erasers.npz', **model['erasers'])
        c = model['checkpoint']
        a = SavedAdversaryPredictor(model['adversaries'], c['training_release'], c['target_mean'], c['target_std'])
        audits = load_audit(prior_seed/key)
        historical = json.loads((prior_seed/key/'validation.json').read_text())
        val = {'A': a.score(views['validation'], data['validation']['y']),
               'D': audit_score(audits, views['validation'], data['validation']['y'])}
        reproduction = {name: {k: val[name][k]['r2']-record[k]['r2'] for k in TARGET_NAMES}
            for name, record in [('A', historical['training_adversary_validation_final_release']),
                                 ('D', historical['validation']['attack']['mlp'])]}
        if max(abs(v) for row in reproduction.values() for v in row.values()) > 1e-6:
            raise ValueError('Original validation predictions not reproduced within dense-map arithmetic tolerance')
        kwargs = {k: CONFIG[k] for k in ('epochs', 'batch_size', 'validation_interval', 'lr')}
        fitted = fit_continuations(model['adversaries'], c['training_release'], c['target_mean'], c['target_std'],
            views['attacker_fit'], data['attacker_fit']['y'], views['validation'], data['validation']['y'],
            seed=seed, out_dir=directory/'fitting', **kwargs)
        predictors = {'A': a, **{name: fitted[name] for name in ('B', 'C')}}
        for name in ('B', 'C'):
            val[name] = predictors[name].score(views['validation'], data['validation']['y'])
        remaining = {k: val['D'][k]['r2']-max(val['B'][k]['r2'], val['C'][k]['r2']) for k in TARGET_NAMES}
        bridge_record = {'eligible': seed == 2 and key == 'E_dual_0.01', 'remaining_validation_gaps': remaining,
                         'trigger_threshold': .05, 'run': False}
        if bridge_record['eligible'] and max(remaining.values()) > .05:
            coordinates = {k: v.clone() for k, v in c['training_release'].items()}
            coordinates['feature_mean'] = torch.tensor(np.stack([v.mean(0) for v in views['attacker_fit'][:2]]), dtype=torch.float32)
            coordinates['feature_std'] = torch.tensor(np.stack([v.std(0).clip(1e-6) for v in views['attacker_fit'][:2]]), dtype=torch.float32)
            ym = torch.tensor(data['attacker_fit']['y'].mean(0), dtype=torch.float32)
            ys = torch.tensor(data['attacker_fit']['y'].std(0), dtype=torch.float32)
            bridge = fit_continuations(model['adversaries'], coordinates, ym, ys,
                views['attacker_fit'], data['attacker_fit']['y'], views['validation'], data['validation']['y'],
                seed=seed, out_dir=directory/'bridge_C_coordinates', kinds=('C',), **kwargs)
            predictors['C_bridge'] = bridge['C']
            val['C_bridge'] = bridge['C'].score(views['validation'], data['validation']['y'])
            bridge_record['run'] = True
        mismatch = {}
        for step, state in model['training_maps'].items():
            released = apply_training_map(raw['validation'], state)
            pred = SavedAdversaryPredictor(model['adversaries'], state, c['target_mean'], c['target_std'])
            mismatch[f'training_map_{step}_own_coordinates'] = pred.score(released, data['validation']['y'])
            # Factor release-map versus standardizer changes while preserving weight semantics.
            mismatch[f'final_map_coordinates_{step}'] = pred.score(views['validation'], data['validation']['y'])
            mismatch[f'training_map_{step}_coordinates_400'] = a.score(released, data['validation']['y'])
            mismatch[f'output_change_{step}'] = [float(np.sqrt(np.mean((released[p]-views['validation'][p])**2))) for p in range(3)]
        after = integrity(model)
        if before != after:
            raise AssertionError('Frozen source state changed')
        label = next((v for v in selections.values() if v['key'] == key), {'diagnostic_only': True})
        result['methods'][key] = {'validation': val, 'historical_reproduction_r2_delta': reproduction,
            'prior_selection_label': label, 'prior_assessment_validation': historical['assessment_validation'],
            'integrity_before': before, 'integrity_after': after, 'frozen_unchanged': True,
            'cache_hashes': hashes, 'eraser_mismatch_validation': mismatch, 'bridge': bridge_record,
            'source_checkpoint_sha256': file_hash(prior_seed/key/'training/final.pt'),
            'source_final_eraser_sha256': file_hash(prior_seed/key/'erasers.npz'),
            'selection': {name: pred.metadata for name, pred in predictors.items() if name != 'A'},
            'independent_audit_reference': {f'{p}_{kind}': probe.metadata for (p, kind), probe in audits.items()}}
        dump(directory/'validation.json', result['methods'][key])
        prepared[key] = (model, before, predictors, audits, views)
    controls = {}
    for key in ('A_oracle', 'exposed_target'):
        audits = load_audit(prior_seed/key)
        y = data['validation']['y']
        h = [y[:, p:p+1] if key == 'A_oracle' else y for p in range(2)]
        h += [np.concatenate(h, axis=1)]
        result['controls'][key] = {'validation': {kind: audit_score(audits, h, y, kind) for kind in ('linear', 'mlp')}}
        controls[key] = audits
    selection_path = out/'selection_before_test.json'
    dump(selection_path, {'test_generated': False, 'test_rng_seed': CONFIG['test_rng_seeds'][str(seed)],
         'methods': {key: {'selection': row['selection'], 'bridge': row['bridge'], 'prior_selection_label': row['prior_selection_label']}
                     for key, row in result['methods'].items()}, 'elapsed_seconds': time.perf_counter()-tick})
    # Sealed boundary: no diagnostic fitting after this point.
    final = diagnostic_test(seed, manifest, selection_path)
    np.savez_compressed(out/'diagnostic_test.npz', **final)
    dump(out/'split_manifest_test.json', {'rng_seed': CONFIG['test_rng_seeds'][str(seed)],
        **{k: {'n': len(v), 'sha256': old.digest_array(v)} for k, v in final.items()}})
    for key, (model, before, predictors, audits, cached) in prepared.items():
        views, _, hashes = cache_releases(model, {'test': final}, out/key/'release_cache')
        h = views['test']
        row = result['methods'][key]
        row['test'] = {name: pred.score(h, final['y']) for name, pred in predictors.items()}
        row['test']['D'] = audit_score(audits, h, final['y'])
        row['test_linear_reference'] = audit_score(audits, h, final['y'], 'linear')
        row['cache_hashes'].update(hashes)
        row['integrity_after_test'] = integrity(model)
        if row['integrity_after_test'] != before:
            raise AssertionError('Frozen state changed during test')
        fresh_val = apply_final([old.encode(model['encoder'], data['validation']['x'], p) for p in range(2)], model['erasers'])
        if not all(np.array_equal(x, y) for x, y in zip(fresh_val, cached['validation'])):
            raise AssertionError('Frozen validation outputs changed')
        row['representative_outputs_unchanged'] = True
        dump(out/key/'metrics.json', row)
    for key, audits in controls.items():
        y = final['y']
        h = [y[:, p:p+1] if key == 'A_oracle' else y for p in range(2)]
        h += [np.concatenate(h, axis=1)]
        result['controls'][key]['test'] = {kind: audit_score(audits, h, y, kind) for kind in ('linear', 'mlp')}
    result['runtime_seconds'] = time.perf_counter()-tick
    dump(out/'metrics.json', result)
    print(f'seed={seed}: complete {result["runtime_seconds"]:.3f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'PROTOCOL.md').exists():
        raise FileNotFoundError('Write protocol before diagnostic fitting')
    sources = ['experiments/run_frozen_release_diagnostic.py', 'experiments/frozen_release_adversaries.py',
        'experiments/run_nonlinear_conflict.py', 'experiments/nonlinear_conflict_training.py',
        'experiments/nonlinear_conflict_probes.py', 'experiments/nonlinear_release_training.py',
        'experiments/run_redesign_conflict.py', 'pcrl/models/lora.py', 'pcrl/training/proxy_lagrangian.py']
    records = {'config.json': CONFIG, 'frozen_source_hashes.json': {s: file_hash(ROOT/s) for s in sources},
               'protocol_sha256.json': file_hash(out/'PROTOCOL.md'),
               'historical_evidence_hashes.json': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()}}
    for name, value in records.items():
        path = out/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen evidence changed: {name}')
        if not path.exists():
            dump(path, value)
    for source in sources:
        path = out/'source_snapshot'/source
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/source).read_bytes())
    invocation = out/('invocation_prepare.json' if args.prepare_only else f'invocation_{"_".join(map(str,args.seeds))}.json')
    if invocation.exists():
        raise FileExistsError('Preserve existing invocation; use a fresh diagnostic directory')
    dump(invocation, {'argv': sys.argv, 'time_ns': time.time_ns()})
    if not args.prepare_only:
        for seed in args.seeds:
            run_seed(seed, out/f'seed_{seed}')


if __name__ == '__main__':
    main()
