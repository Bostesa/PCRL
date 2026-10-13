"""ROOT-authorized, queue-only fixed-income occupation pilot.

No arrays, models, torch or numpy are imported until the protection queue and
immutable direct-trial lock have been checked. One worker runs all four matched
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
from .pilot import ARMS, RECIPE, canonical_sha, validate_recipe, validate_direct_lock
from .wires import corrected_trial_wires, prepare_geometry, forward_role, FrozenIncomeGuard


def _seed(source_seed, purpose):
    return int(hashlib.sha256(f'CIS1-direct-trial|{source_seed}|{purpose}'.encode()).hexdigest()[:8], 16)


def _guard(stage=None):
    from .scheduler import assert_active_from_env
    active = assert_active_from_env(stage=stage)
    if active.get('stage', os.environ.get('PCRL_QUEUE_STAGE')) not in ('admission','protection'):
        raise PermissionError('trainer numerical work requires admission or protection queue')
    return active


def _authorization(spec):
    from .source import load_manifest
    from .admission import read_metadata, sha_metadata
    validate_recipe(spec['recipe'])
    capability = spec['direct_trial_lock']
    if sha_metadata(capability['path']) != capability['sha256']:
        raise PermissionError('ROOT direct-trial lock bytes differ')
    lock = read_metadata(capability['path'])
    validate_direct_lock(lock, seed=spec['seed'], checkpoint=spec['checkpoint'],
                         geometry_sha256=spec['geometry_binding_sha256'])
    if sha_metadata(spec['geometry']['path']) != spec['geometry']['sha256']:
        raise PermissionError('per-seed geometry file bytes differ')
    return load_manifest(spec['manifest'], spec['manifest_sha256'])



class OccupationCritics:
    """Six actual-wire critics; both banks train in every continuation arm."""
    def __init__(self, source_seed, sex, geometry, frozen):
        import torch
        _guard()
        prior = torch.bincount(sex, minlength=2).double() / len(sex)
        if (prior <= 0).any():
            raise ValueError('both sensitive groups required in update folds')
        self.geometry = geometry
        self.frozen = frozen
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
        from task_conditioned_representation.kernels import DiagonalReference
        with torch.no_grad():
            self.frozen.verify()
            wires = corrected_trial_wires(model, fixed_x, self.geometry)
            values = [wires[1]['full'], torch.cat([w['full'] for w in wires], 1)]
            prior = self.references or [None, None]
            self.references = [DiagonalReference.fit(value, old) for value, old in zip(values, prior)]
        for reference in self.references:
            if (not torch.isfinite(reference.mean).all() or not torch.isfinite(reference.scale).all()
                    or (reference.scale <= 0).any() or reference.scale.requires_grad):
                raise ValueError('detached finite positive full-coordinate reference required')
        self.frozen.verify()
        return self.snapshots()

    def snapshots(self):
        return [{key: value.tolist() if hasattr(value, 'tolist') else value
                 for key, value in r.snapshot().items()} for r in self.references]

    def views(self, model, x):
        import torch
        if self.references is None:
            raise RuntimeError('fixed FIT reference absent')
        wires = corrected_trial_wires(model, x, self.geometry)
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
                self.frozen.verify()

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
        self.frozen.verify()
        return {'local1': terms[0], 'pair': terms[1], 'critic_ces': ce_values}

    def state_dict(self):
        return {'critics': [[c.state_dict() for c in bank] for bank in self.banks],
                'optimizers': [[o.state_dict() for o in bank] for bank in self.optimizers],
                'references': self.references, 'prior': self.prior, 'entropy': self.entropy,
                'critic_optimizer_steps': self.optimizer_steps}


def _freeze(model, geometry):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('enc.1.', 'head.1.')))
    return FrozenIncomeGuard(model, geometry)


def _verify_income(model, frozen):
    return frozen.verify()


def _draw(rng, n):
    return rng.sample(range(n), RECIPE['batch_size'])


def _normalization(model, critics, x, occupation, sex, source_seed, private_dir):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .gradient_worker import flat_gradient, gradient_summary
    parameters = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    rng = random.Random(_seed(source_seed, 'normalization'))
    records, tensor_records = [], []
    for draw_number in range(RECIPE['normalization_draws']):
        _guard()
        indices = _draw(rng, len(x))
        terms = critics.terms(model, x[indices], sex[indices])
        ce = F.cross_entropy(corrected_trial_wires(model, x[indices], critics.geometry)[1]['centered_logits'], occupation[indices])
        vectors = {'task_ce': flat_gradient(ce, parameters),
                   'sensitive_full_occupation_plus_pair': flat_gradient(terms['local1'] + terms['pair'], parameters)}
        critics.frozen.verify()
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


def _step_gradients(model, x, occupation, terms, coefficient, arm, geometry):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .gradient_worker import flat_gradient, gradient_summary
    parameters = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    wire = corrected_trial_wires(model, x, geometry)[1]; p = wire['probabilities']
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


def _selection_metrics(model, fit, selected, geometry):
    from .evaluation_worker import useful_metrics
    import numpy as np
    views = forward_role(model, fit['X'], fit['feature_names'], geometry, variant='RAW_HYBRID')
    return {task: useful_metrics(fit[task][selected], views[i]['probability'][selected], model.class_orders[i],
                                np.array([(fit[task] == k).mean() for k in model.class_orders[i]]))
            for i, task in enumerate(('income', 'occupation'))}


def _save(model, source_payload, critics, output_dir, epoch, metadata, metrics):
    import torch
    from .source import write_json
    _guard()
    payload = {key: copy.deepcopy(source_payload[key]) for key in
               ('schema', 'feature_names', 'class_orders', 'source_bindings')}
    payload.update(model={name: value.detach().clone() for name, value in model.state_dict().items()},
                   family='COMPRESSED_INCOME_SAFEGUARD', seed=metadata['source_seed'], epoch=epoch,
                   wire_binding=copy.deepcopy(critics.geometry.binding),
                   wire_contract=copy.deepcopy(RECIPE['wire_contract']), wire_variant=metadata['arm'],
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
    from .evaluation_worker import queue_config
    active, spec, verify, queue_guard = queue_config(spec_path, 'protection')
    from .source import load_role, pinned, write_json
    from .admission import read_metadata
    for key in ('direct_trial_lock','geometry','checkpoint'):
        verify(spec[key]['path'], spec[key]['sha256'])
    if spec.get('operation') != 'compressed_income_safeguard_trial' or spec.get('seed') not in (0, 1):
        raise PermissionError('unregistered pilot operation or source seed')
    queue_guard()
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
    geometry = prepare_geometry(read_metadata(spec['geometry']['path']), expected_sha256=spec['geometry_binding_sha256'],
                                source_seed=spec['seed'], checkpoint_sha256=spec['checkpoint']['sha256'])
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
    baseline_metrics = _selection_metrics(source_model, fit, selected, geometry)
    common = {'source_seed': spec['seed'], 'starting_checkpoint_sha256': spec['checkpoint']['sha256'],
              'direct_trial_lock_sha256': spec['direct_trial_lock']['sha256'], 'recipe_sha256': canonical_sha(RECIPE),
              'update_rows': len(update), 'selection_rows': len(selected),
              'update_count_per_epoch': math.ceil(len(update) / 256),
              'INNER_loaded': False, 'source_training_saw_all_FIT': True,
              'diagnostic_training_proxy': True, 'geometry_binding_sha256': geometry.binding_sha256,
              'income_wire_exact_alias': {'income_tensor_sha256': geometry.binding['head_tensor_sha256'][0],
                                          'geometry_binding_sha256': geometry.binding_sha256,
                                          'canonical_role_forward_batch_size': RECIPE['wire_contract']['role_forward_batch_size']}}
    write_json(output_dir / 'EXPOSURE.json', dict(common, inherited_update_row_indices=update.tolist(),
               inherited_selection_row_indices=selected.tolist(), fixed_reference_update_indices=reference_rows), private=True)
    summaries = {}; coefficient = None; normalization_sha = None
    for arm in ARMS:
        _guard()
        model = copy.deepcopy(source_model); model.eval(); frozen = _freeze(model, geometry)
        critics = OccupationCritics(spec['seed'], sex, geometry, frozen)
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
        statuses = {}; steps = 0; epoch_accounting = []
        with (arm_dir / 'UPDATE_LOG.jsonl').open('x') as log:
            os.chmod(arm_dir / 'UPDATE_LOG.jsonl', 0o600)
            for epoch in range(1, 21):
                _guard()
                reference = critics.refresh(model, x[reference_rows])
                write_json(arm_dir / f'epoch_{epoch}_REFERENCE.json', {'epoch': epoch, 'references': reference}, private=True)
                epoch_counts = {'accepted':0,'shortened':0,'rejected':0,'zero_displacement':0}
                for step_in_epoch in range(common['update_count_per_epoch']):
                    for _ in range(5):
                        rows = _draw(batch_rng, len(x)); critics.update(model, x[rows], sex[rows])
                        _verify_income(model, frozen)
                    rows = _draw(batch_rng, len(x))
                    terms = critics.terms(model, x[rows], sex[rows])
                    all_gradients = _step_gradients(model, x[rows], occupation[rows], terms, coefficient, arm, geometry)
                    _guard()
                    result = occupation_refinement_step(model, x[rows], occupation[rows], terms,
                        arm=arm, coefficient=coefficient, learning_rate=0.01, max_norm=0.01, geometry=geometry, frozen_guard=frozen)
                    _verify_income(model, frozen)
                    epoch_counts['accepted'] += int(result['status']=='ACCEPTED')
                    epoch_counts['shortened'] += int(result['status']=='ACCEPTED' and result['accepted_fraction']<1)
                    epoch_counts['rejected'] += int(result['status']=='REJECTED_ZERO_STEP')
                    epoch_counts['zero_displacement'] += int(result['actual_displacement_norm']==0)
                    steps += 1; statuses[result['status']] = statuses.get(result['status'], 0) + 1
                    result.update(epoch=epoch, step_in_epoch=step_in_epoch, update_number=steps,
                        critic_optimizer_steps=critics.optimizer_steps, critic_ces=terms['critic_ces'],
                        update_indices_sha256=canonical_sha(rows), all_gradient_summary=all_gradients)
                    log.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n'); log.flush()
                epoch_accounting.append(dict(epoch=epoch, proposals=common['update_count_per_epoch'],
                    accepted_fraction=epoch_counts['accepted']/common['update_count_per_epoch'],
                    zero_stall_fraction=epoch_counts['zero_displacement']/common['update_count_per_epoch'], **epoch_counts))
                if epoch in RECIPE['checkpoint_epochs']:
                    _verify_income(model, frozen)
                    checkpoints.append(_save(model, source_payload, critics, arm_dir, epoch,
                                               dict(metadata, model_updates=steps), _selection_metrics(model, fit, selected, geometry)))
        expected = (80 + 5 * steps) * 6
        if critics.optimizer_steps != expected:
            raise AssertionError('matched ancillary critic update budget differs')
        summaries[arm] = dict(model_updates=steps, critic_optimizer_steps=critics.optimizer_steps,
                             update_status_counts=statuses, epoch_accounting=epoch_accounting, checkpoints=checkpoints,
                             update_log_sha256=_sha(arm_dir / 'UPDATE_LOG.jsonl'), **metadata)
        write_json(arm_dir / 'SUMMARY.json', summaries[arm], private=True)
    result = {'schema': 'CIS1-compressed-income-training-v1', 'queue_unit_id': active['unit_id'],
              'recipe': RECIPE, 'arms': summaries,
              'unchanged_RAW_HYBRID_control': {'checkpoint_sha256': spec['checkpoint']['sha256'],
                                       'source_family': 'RAW_JOINT', 'strength': 0.3, 'epoch': 5},
              'coefficient': coefficient, 'seed': spec['seed'], 'INNER_loaded': False,
              'nomination_status': 'UNSELECTED_REQUIRE_COMMON_FIT2_AND_FROZEN_ATT_RULE',
              'scientific_status': 'TRAINING_ARTIFACTS_ONLY_NO_PRIVACY_OR_UTILITY_CLAIM'}
    write_json(output_dir / 'SUMMARY.json', result, private=True)
    write_json(spec['output'], result)
    return result


def engineering_smoke(spec_path):
    """Actual architecture, finite synthetic input; no source fit or Adult access.

    Times 80 warm updates per critic, sixteen normalization draws once, and
    two matched proposals in every arm. The full scientific recipe is never
    replaced by this engineering sample. ROOT queues this admission operation.
    """
    from .evaluation_worker import queue_config
    active, spec, verify, queue_guard = queue_config(spec_path, 'admission')
    if spec.get('operation') != 'compressed_income_finite_smoke':
        raise PermissionError('finite engineering operation required')
    import time
    import numpy as np
    import torch
    from task_conditioned_representation.useful_service import UsefulModel
    from .wires import bind_geometry
    from .gradient_worker import occupation_refinement_step
    from .source import write_json
    torch.set_num_threads(1); torch.manual_seed(917)
    source = UsefulModel([f'x{i}' for i in range(83)]); source.eval()
    x=torch.randn(1024,83); occupation=torch.arange(len(x)).remainder(6); sex=torch.arange(len(x)).remainder(2)
    geometry=prepare_geometry(bind_geometry(source,source_seed=0,checkpoint_sha256='0'*64))
    output_dir=Path(spec['private_output']).resolve()
    if '.private' not in output_dir.parts or output_dir.exists(): raise PermissionError('fresh private smoke directory required')
    output_dir.mkdir(parents=True,mode=0o700)
    records={}; coefficient=None
    for arm in ARMS:
        queue_guard(); model=copy.deepcopy(source); frozen=_freeze(model,geometry)
        critics=OccupationCritics(0,sex,geometry,frozen); critics.refresh(model,x[:256])
        start=time.monotonic(); warm_rng=random.Random(_seed(0,'warm'))
        for _ in range(RECIPE['warm_updates_per_critic']):
            rows=_draw(warm_rng,len(x)); critics.update(model,x[rows],sex[rows])
        warm_seconds=time.monotonic()-start
        normalization_seconds=0
        if coefficient is None:
            start=time.monotonic(); coefficient,_=_normalization(model,critics,x,occupation,sex,0,output_dir)
            normalization_seconds=time.monotonic()-start
        step_rng=random.Random(_seed(0,'model_and_critic_draws')); results=[]; start=time.monotonic()
        for step in range(2):
            for _ in range(RECIPE['critic_updates_per_model_step']):
                rows=_draw(step_rng,len(x)); critics.update(model,x[rows],sex[rows])
            rows=_draw(step_rng,len(x)); terms=critics.terms(model,x[rows],sex[rows])
            result=occupation_refinement_step(model,x[rows],occupation[rows],terms,arm=arm,coefficient=coefficient,
                                              geometry=geometry,frozen_guard=frozen,learning_rate=.01,max_norm=.01)
            results.append(result)
        elapsed=time.monotonic()-start
        frozen.verify()
        if critics.optimizer_steps != (80+2*5)*6:raise AssertionError('finite matched critic accounting differs')
        records[arm]={'warm_seconds':warm_seconds,'normalization_seconds':normalization_seconds,
                      'two_proposals_seconds':elapsed,'critic_optimizer_steps':critics.optimizer_steps,'steps':results}
    result={'schema':'CIS1-finite-trainer-smoke-v1','queue_unit_id':active['unit_id'],
            'finite_engineering_only':True,'Adult_roles_loaded':False,'science_fit_count':0,
            'coefficient':coefficient,'arms':records,'recipe_sha256':canonical_sha(RECIPE)}
    write_json(output_dir/'SUMMARY.json',result,private=True);write_json(spec['output'],result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spec');group.add_argument('--engineering-spec')
    args=parser.parse_args()
    engineering_smoke(args.engineering_spec) if args.engineering_spec else run(args.spec)
