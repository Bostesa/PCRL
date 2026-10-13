"""ROOT-authorized, queue-only fixed-income occupation pilot.

No arrays, models, torch or numpy are imported until the protection queue and
immutable root trigger have been checked. One worker runs all four matched
arms for a source seed. No INNER data or ATT fit enters this trainer.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from .pilot import ARMS, RECIPE, canonical_sha, validate_recipe


def _seed(source_seed, purpose):
    return int(hashlib.sha256(f'OB1-pilot|{source_seed}|{purpose}'.encode()).hexdigest()[:8], 16)


def _guard():
    from .scheduler import assert_active_from_env
    return assert_active_from_env(stage='protection')


def _authorization(spec):
    from .source import load_manifest
    from .admission import read_metadata, sha_metadata
    validate_recipe(spec['recipe'])
    trigger = spec['trigger_lock']
    if sha_metadata(trigger['path']) != trigger['sha256']:
        raise PermissionError('ROOT pilot trigger bytes differ')
    lock = read_metadata(trigger['path'])
    if (lock.get('owner') != 'ROOT' or lock.get('status') != 'PILOT_TRIGGERED'
            or lock.get('INNER_used_for_trigger') is not False
            or lock.get('recipe_sha256') != canonical_sha(RECIPE)):
        raise PermissionError('explicit FIT/ATT-only root pilot trigger required')
    if not lock.get('diagnosis_sha256'):
        raise PermissionError('pilot trigger must bind the concrete diagnosis')
    start = lock['starting_checkpoints'][str(spec['seed'])]
    if start != spec['checkpoint']:
        raise PermissionError('pilot starting state not bound by ROOT trigger')
    if set(lock['starting_checkpoints']) != {'0', '1'}:
        raise PermissionError('both source-seed starting bindings required')
    return load_manifest(spec['manifest'], spec['manifest_sha256'])


class OccupationCritics:
    """Six actual-wire critics; both banks train in every continuation arm."""
    def __init__(self, source_seed, sex):
        import torch
        _guard()
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        if (prior <= 0).any():
            raise ValueError('both sensitive groups required in update folds')
        self.prior = prior
        self.entropy = float(-(prior * prior.log()).sum())
        self.banks, self.optimizers, self.references = [], [], None
        self.optimizer_steps = 0
        for view, width in enumerate((29, 50)):
            bank, optimizers = [], []
            for kind in RECIPE['critic_kinds']:
                torch.manual_seed(_seed(source_seed, f'critic|{view}|{kind}'))
                if kind == 'linear':
                    critic = torch.nn.Linear(width, 2, dtype=torch.float64)
                elif kind == 'mlp32':
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 32, dtype=torch.float64),
                                                torch.nn.ReLU(), torch.nn.Linear(32, 2, dtype=torch.float64))
                else:
                    critic = torch.nn.Sequential(torch.nn.Linear(width, 64, dtype=torch.float64),
                        torch.nn.ReLU(), torch.nn.Linear(64, 64, dtype=torch.float64),
                        torch.nn.ReLU(), torch.nn.Linear(64, 2, dtype=torch.float64))
                bank.append(critic)
                optimizers.append(torch.optim.Adam(critic.parameters(), lr=RECIPE['critic_learning_rate']))
            self.banks.append(bank); self.optimizers.append(optimizers)

    def refresh(self, model, fixed_x):
        import torch
        from task_conditioned_representation.models import actual_wires
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            wires = actual_wires(model, fixed_x)
            values = [wires[1]['full'], torch.cat([w['full'] for w in wires], 1)]
            prior = self.references or [None, None]
            self.references = [DiagonalReference.fit(value, old) for value, old in zip(values, prior)]
        for reference in self.references:
            if (not torch.isfinite(reference.mean).all() or not torch.isfinite(reference.scale).all()
                    or (reference.scale <= 0).any() or reference.scale.requires_grad):
                raise ValueError('detached finite positive full-coordinate reference required')
        return self.snapshots()

    def snapshots(self):
        return [{key: value.tolist() if hasattr(value, 'tolist') else value
                 for key, value in r.snapshot().items()} for r in self.references]

    def views(self, model, x):
        import torch
        from task_conditioned_representation.models import actual_wires
        if self.references is None:
            raise RuntimeError('fixed FIT reference absent')
        wires = actual_wires(model, x)
        values = [wires[1]['full'], torch.cat([w['full'] for w in wires], 1)]
        return [r.normalize(value) for r, value in zip(self.references, values)]

    def update(self, model, x, sex):
        import torch
        import torch.nn.functional as F
        _guard()
        with torch.no_grad():
            views = self.views(model, x)
        for bank, optimizers, view in zip(self.banks, self.optimizers, views):
            for critic, optimizer in zip(bank, optimizers):
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(critic(view), sex)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite critic loss')
                loss.backward(); optimizer.step(); self.optimizer_steps += 1

    def terms(self, model, x, sex):
        import torch
        import torch.nn.functional as F
        views = self.views(model, x)
        terms, ce_values = [], []
        for bank, view in zip(self.banks, views):
            for critic in bank:
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
            losses = torch.stack([F.cross_entropy(c(view), sex) for c in bank]
                                 + [view.sum() * 0 + self.entropy])
            terms.append(1 - losses.min() / self.entropy)
            ce_values.append(losses.detach().tolist())
        return {'local1': terms[0], 'pair': terms[1], 'critic_ces': ce_values}

    def state_dict(self):
        return {'critics': [[c.state_dict() for c in bank] for bank in self.banks],
                'optimizers': [[o.state_dict() for o in bank] for bank in self.optimizers],
                'references': self.references, 'prior': self.prior, 'entropy': self.entropy,
                'critic_optimizer_steps': self.optimizer_steps}


def _freeze(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('enc.1.', 'head.1.')))
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if name.startswith(('enc.0.', 'head.0.'))}


def _verify_income(model, frozen):
    import torch
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()):
        raise AssertionError('frozen income state changed')


def _draw(rng, n):
    return rng.sample(range(n), RECIPE['batch_size'])


def _normalization(model, critics, x, occupation, sex, source_seed, private_dir):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .gradient_worker import flat_gradient, gradient_summary
    from task_conditioned_representation.models import actual_wires
    parameters = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    rng = random.Random(_seed(source_seed, 'normalization'))
    records, tensor_records = [], []
    for draw_number in range(RECIPE['normalization_draws']):
        _guard()
        indices = _draw(rng, len(x))
        terms = critics.terms(model, x[indices], sex[indices])
        ce = F.cross_entropy(actual_wires(model, x[indices])[1]['centered_logits'], occupation[indices])
        vectors = {'task_ce': flat_gradient(ce, parameters),
                   'sensitive_full_occupation_plus_pair': flat_gradient(terms['local1'] + terms['pair'], parameters)}
        records.append(dict(draw=draw_number, summary=gradient_summary(vectors)))
        tensor_records.append({'indices': indices, **{k: torch.from_numpy(v) for k, v in vectors.items()}})
    task = float(np.median([r['summary']['norms']['task_ce'] for r in records]))
    sensitive = float(np.median([r['summary']['norms']['sensitive_full_occupation_plus_pair'] for r in records]))
    path = private_dir / 'NORMALIZATION_TENSORS.pt'
    torch.save(tensor_records, path); path.chmod(0o600)
    coefficient = RECIPE['normalization_target_ratio'] * task / sensitive if sensitive > 0 else None
    receipt = {'task_median_norm': task, 'sensitive_median_norm': sensitive,
               'coefficient': coefficient, 'target_ratio': 3.0, 'draws': records,
               'normalization_tensor_sha256': _sha(path),
               'parameter_order': [name for name, p in model.named_parameters() if p.requires_grad]}
    valid = (math.isfinite(task) and math.isfinite(sensitive)
             and min(task, sensitive) > RECIPE['normalization_min_median_norm']
             and coefficient is not None and math.isfinite(coefficient)
             and coefficient <= RECIPE['maximum_coefficient'])
    receipt['status'] = 'FIXED' if valid else 'UNAVAILABLE_FIXED_NORMALIZATION'
    from .source import write_json
    write_json(private_dir / 'NORMALIZATION.json', receipt, private=True)
    if not valid:
        raise ValueError('fixed normalization unavailable; preserve receipt and do not tune recipe')
    return coefficient, receipt


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _step_gradients(model, x, occupation, terms, coefficient, arm):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .gradient_worker import flat_gradient, gradient_summary
    from task_conditioned_representation.models import actual_wires
    parameters = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    wire = actual_wires(model, x)[1]; p = wire['probabilities']
    ce = F.cross_entropy(wire['centered_logits'], occupation)
    nll = -p[torch.arange(len(x)), occupation].clamp_min(1e-12).log().mean()
    brier = (p - F.one_hot(occupation, 6).to(torch.float64)).square().sum(1).mean()
    # Even TASK_ONLY reports the ancillary privacy gradients; its applied
    # model term is zero, so disabling privacy remains explicit in the log.
    vectors = {'task_ce': flat_gradient(ce, parameters),
               'occupation_nll': flat_gradient(nll, parameters),
               'occupation_brier': flat_gradient(brier, parameters),
               'sensitive_local1': flat_gradient(terms['local1'], parameters),
               'sensitive_pair': flat_gradient(terms['pair'], parameters)}
    vectors['sensitive_full'] = vectors['sensitive_local1'] + vectors['sensitive_pair']
    applied = np.zeros_like(vectors['task_ce']) if arm == 'TASK_ONLY' else (
        vectors['sensitive_local1'] if arm == 'LOCAL_TWIN' else vectors['sensitive_full'])
    vectors['sensitive_model_term'] = applied
    proposal = -.01 * (vectors['task_ce'] + coefficient * applied)
    return dict(gradient_summary(vectors), proposed_SGD_displacement_norm=float(np.linalg.norm(proposal)),
                privacy_gradient_disabled=arm == 'TASK_ONLY')


def _selection_metrics(model, fit, selected):
    from .evaluation_worker import useful_metrics
    import numpy as np
    views = model.route_raw(fit['X'][selected], fit['feature_names'])
    return {task: useful_metrics(fit[task][selected], views[i]['probability'], model.class_orders[i],
                                np.array([(fit[task] == k).mean() for k in model.class_orders[i]]))
            for i, task in enumerate(('income', 'occupation'))}


def _save(model, source_payload, critics, output_dir, epoch, metadata, metrics):
    import torch
    from .source import write_json
    _guard()
    payload = {key: copy.deepcopy(source_payload[key]) for key in
               ('schema', 'feature_names', 'class_orders', 'source_bindings')}
    payload.update(model={name: value.detach().clone() for name, value in model.state_dict().items()},
                   family='OCCUPATION_BOUNDED_PILOT', seed=metadata['source_seed'], epoch=epoch,
                   recipe=copy.deepcopy(RECIPE), coefficient=metadata['coefficient'],
                   objective_state=critics.state_dict(), pilot_metadata=dict(metadata, epoch=epoch),
                   optimizer={'type': 'SGD', 'momentum': 0.0, 'learning_rate': 0.01},
                   selection_history=metrics)
    path = output_dir / f'epoch_{epoch}.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream)
    path.chmod(0o600)
    result = {'epoch': epoch, 'checkpoint_sha256': _sha(path), 'metrics': metrics,
              'frozen_income_bitwise_unchanged': True}
    write_json(output_dir / f'epoch_{epoch}_METRICS.json', result, private=True)
    return result


def run(spec_path):
    active = _guard()
    from .source import load_role, pinned, write_json
    from .admission import read_metadata
    spec = read_metadata(spec_path)
    if spec.get('operation') != 'occupation_bounded_pilot' or spec.get('seed') not in (0, 1):
        raise PermissionError('unregistered pilot operation or source seed')
    manifest = _authorization(spec)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .gradient_worker import occupation_refinement_step
    from .gradient_worker import QP_TOL, FINITE_TOL, BACKTRACK, NORM_REPAIR_SAFETY, NORM_REPAIR_ATTEMPTS
    if (QP_TOL != 1e-10 or FINITE_TOL != 1e-10 or list(BACKTRACK) != RECIPE['backtrack_fractions']
            or NORM_REPAIR_SAFETY != RECIPE['actual_norm_repair_safety_fraction']
            or NORM_REPAIR_ATTEMPTS != RECIPE['actual_norm_repair_maximum_attempts']):
        raise PermissionError('gradient guard implementation differs from frozen recipe')
    source_model, source_payload = load_checkpoint(pinned(spec['checkpoint']))
    if (source_payload.get('family') != 'RAW_JOINT' or source_payload.get('strength') != 0.3
            or source_payload.get('epoch') != 5 or source_payload.get('seed') != spec['seed']):
        raise PermissionError('starting package must be admitted RAW strength .3 epoch 5')
    fit = load_role(manifest, spec['seed'], 'OSF_DEFENSE_FIT',
                    {'features', 'occupation', 'sex', 'income'})
    if tuple(source_model.names) != tuple(map(str, fit['feature_names'])):
        raise PermissionError('pilot feature order differs from admitted source')
    update = np.flatnonzero(np.isin(fit['fold'], [0, 1])); selected = np.flatnonzero(fit['fold'] == 2)
    if len(update) < 256 or not len(selected):
        raise PermissionError('inherited update/selection folds unavailable')
    x = torch.from_numpy(fit['X'][update]); occupation = torch.from_numpy(fit['occupation'][update]).long()
    sex = torch.from_numpy(fit['sex'][update]).long()
    output_dir = Path(spec['private_output']).resolve()
    if '.private' not in output_dir.parts or output_dir.exists():
        raise PermissionError('fresh private pilot directory required')
    output_dir.mkdir(parents=True, mode=0o700)
    reference_rows = _draw(random.Random(_seed(spec['seed'], 'reference')), len(x))
    baseline_metrics = _selection_metrics(source_model, fit, selected)
    common = {'source_seed': spec['seed'], 'starting_checkpoint_sha256': spec['checkpoint']['sha256'],
              'trigger_sha256': spec['trigger_lock']['sha256'], 'recipe_sha256': canonical_sha(RECIPE),
              'update_rows': len(update), 'selection_rows': len(selected),
              'update_count_per_epoch': math.ceil(len(update) / 256),
              'INNER_loaded': False, 'source_training_saw_all_FIT': True,
              'diagnostic_training_proxy': True, 'income_wire_exact_alias': spec['checkpoint']['sha256']}
    write_json(output_dir / 'EXPOSURE.json', dict(common, inherited_update_row_indices=update.tolist(),
               inherited_selection_row_indices=selected.tolist(), fixed_reference_update_indices=reference_rows), private=True)
    summaries = {}; coefficient = None; normalization_sha = None
    for arm in ARMS:
        _guard()
        model = copy.deepcopy(source_model); model.eval(); frozen = _freeze(model)
        critics = OccupationCritics(spec['seed'], sex)
        reference = critics.refresh(model, x[reference_rows])
        warm_rng = random.Random(_seed(spec['seed'], 'warm'))
        for _ in range(80):
            rows = _draw(warm_rng, len(x)); critics.update(model, x[rows], sex[rows])
            _verify_income(model, frozen)
        arm_dir = output_dir / arm; arm_dir.mkdir(mode=0o700)
        if coefficient is None:
            coefficient, receipt = _normalization(model, critics, x, occupation, sex, spec['seed'], output_dir)
            normalization_sha = canonical_sha(receipt)
        metadata = dict(common, arm=arm, coefficient=coefficient, normalization_receipt_sha256=normalization_sha,
                        warm_optimizer_steps=critics.optimizer_steps)
        checkpoints = [_save(model, source_payload, critics, arm_dir, 0, metadata, baseline_metrics)]
        batch_rng = random.Random(_seed(spec['seed'], 'model_and_critic_draws'))
        statuses = {}; steps = 0
        with (arm_dir / 'UPDATE_LOG.jsonl').open('x') as log:
            os.chmod(arm_dir / 'UPDATE_LOG.jsonl', 0o600)
            for epoch in range(1, 21):
                _guard()
                reference = critics.refresh(model, x[reference_rows])
                write_json(arm_dir / f'epoch_{epoch}_REFERENCE.json', {'epoch': epoch, 'references': reference}, private=True)
                for step_in_epoch in range(common['update_count_per_epoch']):
                    for _ in range(5):
                        rows = _draw(batch_rng, len(x)); critics.update(model, x[rows], sex[rows])
                        _verify_income(model, frozen)
                    rows = _draw(batch_rng, len(x))
                    terms = critics.terms(model, x[rows], sex[rows])
                    all_gradients = _step_gradients(model, x[rows], occupation[rows], terms, coefficient, arm)
                    _guard()
                    result = occupation_refinement_step(model, x[rows], occupation[rows], terms,
                        arm=arm, coefficient=coefficient, learning_rate=0.01, max_norm=0.01)
                    _verify_income(model, frozen)
                    steps += 1; statuses[result['status']] = statuses.get(result['status'], 0) + 1
                    result.update(epoch=epoch, step_in_epoch=step_in_epoch, update_number=steps,
                        critic_optimizer_steps=critics.optimizer_steps, critic_ces=terms['critic_ces'],
                        update_indices_sha256=canonical_sha(rows), all_gradient_summary=all_gradients)
                    log.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n'); log.flush()
                if epoch in RECIPE['checkpoint_epochs']:
                    _verify_income(model, frozen)
                    checkpoints.append(_save(model, source_payload, critics, arm_dir, epoch,
                                               dict(metadata, model_updates=steps), _selection_metrics(model, fit, selected)))
        expected = (80 + 5 * steps) * 6
        if critics.optimizer_steps != expected:
            raise AssertionError('matched ancillary critic update budget differs')
        summaries[arm] = dict(model_updates=steps, critic_optimizer_steps=critics.optimizer_steps,
                             update_status_counts=statuses, checkpoints=checkpoints,
                             update_log_sha256=_sha(arm_dir / 'UPDATE_LOG.jsonl'), **metadata)
        write_json(arm_dir / 'SUMMARY.json', summaries[arm], private=True)
    result = {'schema': 'OB1-occupation-pilot-training-v1', 'queue_unit_id': active['unit_id'],
              'recipe': RECIPE, 'arms': summaries,
              'unchanged_RAW_control': {'checkpoint_sha256': spec['checkpoint']['sha256'],
                                       'source_family': 'RAW_JOINT', 'strength': 0.3, 'epoch': 5},
              'coefficient': coefficient, 'seed': spec['seed'], 'INNER_loaded': False,
              'nomination_status': 'UNSELECTED_REQUIRE_COMMON_FIT2_AND_FROZEN_ATT_RULE',
              'scientific_status': 'TRAINING_ARTIFACTS_ONLY_NO_PRIVACY_OR_UTILITY_CLAIM'}
    write_json(output_dir / 'SUMMARY.json', result, private=True)
    write_json(spec['output'], result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
