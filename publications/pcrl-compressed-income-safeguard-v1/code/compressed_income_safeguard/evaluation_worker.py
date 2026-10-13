"""Queue-owned forwarding, FIT-fold2 trajectory utility, and frozen readers."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def queue_config(spec_path, stage):
    """Bind every byte read by a numerical worker to the new durable queue."""
    from .scheduler import assert_active_from_env
    active = assert_active_from_env(stage=stage)
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    unit = state['units'][active['unit_id']]
    registered = {str(Path(i['path']).resolve()): i['sha256']
                  for i in unit['spec']['sources']}
    def verify(path, expected):
        p = Path(path)
        if p.is_symlink() or not expected or registered.get(str(p.resolve())) != expected:
            raise PermissionError('source absent from active occupation queue')
        # INNER truth must not be hashed before all reporting dependencies freeze.
        low = p.as_posix().lower()
        from .admission import inner_truth_path
        if inner_truth_path(p):
            raise PermissionError('INNER truth requires reporting-specific admission')
        with p.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise PermissionError('active queue source hash differs')
        return p
    p = Path(spec_path)
    verify(p, registered.get(str(p.resolve())))
    cfg = json.loads(p.read_text())
    if cfg['unit_id'] != active['unit_id']:
        raise PermissionError('worker and queue unit differ')
    def guard():
        assert_active_from_env(unit_id=active['unit_id'], stage=stage)
        verify(p, registered[str(p.resolve())])
        for name in ('science_lock', 'diagnostic_lock'):
            if name in cfg:
                verify(cfg[name], cfg[name+'_sha256'])
    guard()
    return active, cfg, verify, guard


def manifest_and_model(cfg, verify):
    from .source import load_manifest
    from task_conditioned_representation.models import load_checkpoint
    manifest = load_manifest(verify(cfg['manifest'], cfg['manifest_sha256']), cfg['manifest_sha256'])
    model, payload = load_checkpoint(verify(cfg['checkpoint'], cfg['checkpoint_sha256']), cfg['checkpoint_sha256'])
    if payload.get('seed') != cfg['seed']:
        raise PermissionError('checkpoint source seed differs')
    return manifest, model


def geometry_from_cfg(cfg, verify):
    from .wires import prepare_geometry
    item=cfg['geometry']
    binding=json.loads(verify(item['path'],item['sha256']).read_text())
    return prepare_geometry(binding,expected_sha256=cfg['geometry_binding_sha256'],source_seed=cfg['seed'])


def forward_release(model, role, *, include_sex=False, geometry=None, variant=None):
    """Literal canonical per-recipient wire; labels never enter the adapter."""
    from .wires import forward_role, VARIANTS
    if geometry is None or variant not in VARIANTS:
        raise PermissionError('explicit corrected per-recipient wire contract required')
    import numpy as np
    views=forward_role(model,role['X'],role['feature_names'],geometry,variant=variant)
    out={k:role[k] for k in ('row_id','unit')}
    out.update(r1=views[0]['wire'],r2=views[1]['wire'])
    if include_sex:out['sex']=role['sex']
    for key in ('r1','r2'):
        for group in np.unique(out['unit']):
            ix=np.flatnonzero(out['unit']==group)
            if len(ix)>1 and not np.array_equal(out[key][ix],np.repeat(out[key][ix[:1]],len(ix),axis=0)):
                raise PermissionError('duplicate group canonical releases differ')
    return out

def useful_metrics(y, probability, class_order, constant_probability):
    """Inherited metric definitions, with the inherited FIT constant reference.

    A class missing from the selection fold has unavailable recall/confidence;
    fixed-list balanced accuracy retains the predecessor zero-recall rule.
    """
    import numpy as np
    from task_conditioned_representation.inference import task_metrics, task_vectors
    y = np.asarray(y); p = np.asarray(probability, dtype=np.float64)
    order = np.asarray(class_order)
    out = task_metrics(y, p, order)
    constant = np.asarray(constant_probability, dtype=np.float64)
    if constant.shape != (len(order),):
        raise ValueError('constant class list differs')
    baseline = task_vectors(y, np.tile(constant, (len(y), 1)), order)
    out['constant_prediction'] = {key: float(value.mean()) for key, value in baseline.items()}
    out['gain_over_constant'] = {'accuracy': out['accuracy']-out['constant_prediction']['accuracy'],
                                 'nll': out['constant_prediction']['nll']-out['nll'],
                                 'brier': out['constant_prediction']['brier']-out['brier']}
    out['confidence_by_true_class'] = [None if not np.any(y == k) else float(p[y == k].max(1).mean()) for k in order]
    out['true_class_probability_by_class'] = [None if not np.any(y == k) else float(p[y == k, j].mean()) for j, k in enumerate(order)]
    out['per_class_recall_available'] = [bool(np.any(y == k)) for k in order]
    out['constant_reference'] = 'class prior over all inherited FIT, source-training exposure acknowledged'
    return out


def trajectory_worker(spec_path):
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .source import load_role
    from task_conditioned_representation.inference import point_utility
    from task_conditioned_representation.utility_worker import reference_probability_key
    from task_conditioned_representation.attacks import save_npz, write_json
    manifest, model = manifest_and_model(cfg, verify)
    fit = load_role(manifest, cfg['seed'], 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation', 'references'})
    selected = np.flatnonzero(fit['fold'] == 2)
    if not len(selected):
        raise PermissionError('inherited FIT selection fold empty')
    guard()
    from .wires import forward_role
    geometry=geometry_from_cfg(cfg,verify)
    whole=forward_role(model,fit['X'],fit['feature_names'],geometry,variant=cfg['service_variant'])
    views=[{k:v[selected] for k,v in w.items()} for w in whole]
    arrays = {key: fit[key][selected] for key in ('row_id', 'unit', 'income', 'occupation')}
    records = {}
    for i, task in enumerate(('income', 'occupation')):
        order = model.class_orders[i]
        prior = np.array([(fit[task] == value).mean() for value in order])
        metrics = useful_metrics(fit[task][selected], views[i]['probability'], order, prior)
        refs = {}
        arrays[f"s{cfg['seed']}__{cfg['arm_id']}__p{i+1}"] = views[i]['probability']
        for ref in ('U0', 'Ucal'):
            probability = fit['references'][f'{ref}_{i+1}'][selected]
            baseline = useful_metrics(fit[task][selected], probability, order, prior)
            refs[ref] = {'metrics': baseline, 'difference': point_utility(metrics, baseline)}
            arrays[reference_probability_key(cfg['seed'], ref, i+1)] = probability
        records[task] = {'metrics': metrics, 'references': refs}
    root = Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():
        raise PermissionError('fresh private trajectory directory required')
    root.mkdir(parents=True, mode=0o700)
    inputs_sha = save_npz(root/'FIT_SELECTION_UTILITY.npz', **arrays)
    out = {'schema': 'OB1-FIT-fold2-utility-v1', 'unit_id': active['unit_id'],
           'arm_id': cfg['arm_id'], 'seed': cfg['seed'], 'checkpoint_sha256': cfg['checkpoint_sha256'],
           'selection_rows': len(selected), 'selection_groups': len(np.unique(fit['unit'][selected])),
           'metrics': records, 'utility_input_sha256': inputs_sha,
           'source_training_saw_all_FIT': True, 'diagnostic_training_proxy': True,
           'original_caps_display_only': {'accuracy_loss': .010, 'nll_excess': .010, 'brier_excess': .005},
           'direct_FIT_reference_confidence_gate_applied': False,
           'assessment_feasibility_established': False, 'INNER_labels_opened_or_hashed': False}
    write_json(root/'UTILITY_COMPLETE.json', out); write_json(cfg['public_output'], out, False)
    return out


def fit_worker(spec_path):
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    import torch
    torch.set_num_threads(1)
    from .source import load_role
    from .readers import fit_development_bank
    from task_conditioned_representation.attacks import write_json, sha_file
    manifest, model = manifest_and_model(cfg, verify)
    att = load_role(manifest, cfg['seed'], 'ATTACK_FIT_NEW', {'features', 'sex'})
    inner = None
    if cfg.get('forward_inner', True):
        inner = load_role(manifest, cfg['seed'], 'INNER_SELECTION', {'features'})
    diagnostic = None
    if cfg.get('forward_fit_selection', True):
        import numpy as np
        fit = load_role(manifest, cfg['seed'], 'OSF_DEFENSE_FIT', {'features'})
        ix = np.flatnonzero(fit['fold'] == 2)
        diagnostic = {**fit, 'X': fit['X'][ix], 'row_id': fit['row_id'][ix], 'unit': fit['unit'][ix]}
    aliases = cfg.get('view_aliases')
    if aliases:
        for alias in aliases.values():
            # All model/freeze sources are registered and pinned by this unit.
            for pin in alias['sources']:
                verify(pin['path'], pin['sha256'])
    geometry=geometry_from_cfg(cfg,verify)
    # FIT2 must slice a full-role canonical forward, not change its batching.
    diagnostic_release=None
    if cfg.get('forward_fit_selection',True):
        full=forward_release(model,fit,geometry=geometry,variant=cfg['service_variant'])
        diagnostic_release={key:value[ix] for key,value in full.items()}
    receipt=fit_development_bank(
        forward_release(model,att,include_sex=True,geometry=geometry,variant=cfg['service_variant']),
        None if inner is None else forward_release(model,inner,geometry=geometry,variant=cfg['service_variant']),
        seed=cfg['seed'],private_output=cfg['private_output'],queue_guard=guard,
        bindings=cfg['bindings'],include_base=True,include_extended=True,
        controls=cfg.get('controls',False),view_aliases=aliases,diagnostic=diagnostic_release)
    out = {**receipt, 'unit_id': active['unit_id'], 'arm_id': cfg['arm_id'],
           'frozen_complete_sha256': sha_file(Path(cfg['private_output'])/'FROZEN_COMPLETE.json')}
    write_json(cfg['public_output'], out, False)
    return out


def require_reporting(cfg, verify):
    """Verify global committed freeze before even hashing an INNER truth pin."""
    from .source import _reporting_seal
    seal = cfg['reporting_seal']
    verify(seal['path'], seal['sha256'])
    lock = _reporting_seal(seal)
    pins = seal.get('reader_freezes', [])
    if not pins:
        raise PermissionError('global reporting freeze needs every reader bank pin')
    if sorted(p['sha256'] for p in pins) != sorted(lock.get('reader_freeze_sha256', [])):
        raise PermissionError('reader freeze list differs from committed reporting hashes')
    for pin in pins:
        receipt = json.loads(verify(pin['path'], pin['sha256']).read_text())
        if receipt.get('status') != 'FROZEN_COMPLETE':
            raise PermissionError('global reporting reader dependency is not frozen')
    return seal, lock


def report_worker(spec_path):
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    import numpy as np
    from .source import load_manifest, load_role
    from .readers import score_envelopes
    from task_conditioned_representation.attacks import write_json, score_bank
    manifest = load_manifest(verify(cfg['manifest'], cfg['manifest_sha256']), cfg['manifest_sha256'])
    freeze = json.loads(verify(cfg['freeze']['path'], cfg['freeze']['sha256']).read_text())
    if freeze.get('status') != 'FROZEN_COMPLETE' or freeze.get('source_seed') != cfg['seed']:
        raise PermissionError('report bank identity differs')
    for name, field in (('release', 'inner_release_sha256'), ('scores', 'score_sha256')):
        verify(cfg[name]['path'], cfg[name]['sha256'])
        if freeze.get(field) != cfg[name]['sha256']:
            raise PermissionError('report score/release differs from frozen bank')
    with np.load(cfg['release']['path'], allow_pickle=False) as z:
        if set(z.files) != {'row_id', 'unit', 'r1', 'r2'}:
            raise PermissionError('report reader release contains forbidden label inputs')
        identities = {key: z[key].copy() for key in ('row_id', 'unit')}
    truth = load_role(manifest, cfg['seed'], 'INNER_SELECTION', {'features', 'sex'}, reporting_seal=seal)
    if any(not np.array_equal(identities[key], truth[key]) for key in identities):
        raise PermissionError('INNER label and frozen score people differ')
    with np.load(cfg['scores']['path'], allow_pickle=False) as z:
        scores = {key: z[key].copy() for key in z.files}
    result = score_envelopes(scores, truth['sex'])
    controls = None
    if all(f'{view}__{name}' in scores for view in ('r1','r2','pair') for name in ('SHUFFLED','PLANTED_0','PLANTED_1')):
        # Mature real-data controls use the inherited shuffled protected labels.
        # Restrict its envelope to the declared predecessor primary slate.
        from .readers import READERS
        base = {name: value for name, value in scores.items() if name.split('__')[-1] in READERS or name.split('__',1)[1].startswith(('SHUFFLED','PLANTED'))}
        controls = score_bank(base, truth['sex'], truth['unit'])
    out = {'schema': 'OB1-frozen-INNER-recovery-v1', 'unit_id': active['unit_id'],
           'arm_id': cfg['arm_id'], 'seed': cfg['seed'], 'views': result,
           'controls': controls, 'freeze_sha256': cfg['freeze']['sha256'],
           'scores_sha256': cfg['scores']['sha256'], 'reporting_lock_sha256': seal['sha256'],
           'exploratory_assessment': True, 'post_assessment_selection_allowed': False}
    write_json(cfg['public_output'], out, False)
    return out


def report_utility_worker(spec_path):
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .source import load_role
    from task_conditioned_representation.inference import task_metrics, point_utility
    from task_conditioned_representation.utility_worker import reference_probability_key
    from task_conditioned_representation.attacks import save_npz, write_json
    manifest, model = manifest_and_model(cfg, verify)
    truth = load_role(manifest, cfg['seed'], 'INNER_SELECTION', {'features','income','occupation','references'}, reporting_seal=seal)
    from .wires import forward_role
    wire=forward_role(model,truth['X'],truth['feature_names'],geometry_from_cfg(cfg,verify),variant=cfg['service_variant'])
    arrays = {key: truth[key] for key in ('row_id','unit','income','occupation')}
    tasks = {}; clauses = []
    for i, task in enumerate(('income','occupation')):
        metrics = task_metrics(truth[task], wire[i]['probability'], model.class_orders[i])
        arrays[f"s{cfg['seed']}__{cfg['arm_id']}__p{i+1}"] = wire[i]['probability']
        refs = {}
        for ref in ('U0','Ucal'):
            p = truth['references'][f'{ref}_{i+1}']
            arrays[reference_probability_key(cfg['seed'], ref, i+1)] = p
            baseline = task_metrics(truth[task], p, model.class_orders[i]); difference = point_utility(metrics, baseline)
            refs[ref] = {'metrics': baseline, 'difference': difference}
            for name, cap in (('accuracy_loss',.010),('nll_excess',.010),('brier_excess',.005)):
                clauses.append({'task': task, 'reference': ref, 'metric': name,
                                'value': difference[name], 'cap': cap, 'pass': difference[name] <= cap})
        tasks[task] = {'metrics': metrics, 'references': refs}
    root = Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():
        raise PermissionError('fresh private INNER utility output required')
    root.mkdir(parents=True, mode=0o700)
    sha = save_npz(root/'INNER_UTILITY_INPUTS.npz', **arrays)
    out = {'schema': 'OB1-original-INNER-point-utility-v1', 'unit_id': active['unit_id'],
           'arm_id': cfg['arm_id'], 'seed': cfg['seed'], 'tasks': tasks, 'clauses': clauses,
           'point_eligible_seed': all(clause['pass'] for clause in clauses),
           'passed': sum(clause['pass'] for clause in clauses), 'total': 12,
           'utility_input_sha256': sha, 'reporting_lock_sha256': seal['sha256'],
           'reference_namespace': 'REF_U0/REF_Ucal distinct from candidate U0',
           'exploratory_assessment': True}
    write_json(cfg['public_output'], out, False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('trajectory', 'fit', 'report', 'utility-report'), required=True)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    out = {'trajectory': trajectory_worker, 'fit': fit_worker, 'report': report_worker, 'utility-report': report_utility_worker}[args.phase](args.spec)
    print(json.dumps({'status': out.get('status', 'DIAGNOSTIC_COMPLETE'), 'arm_id': out['arm_id'], 'seed': out.get('seed', out.get('source_seed'))}))

if __name__ == '__main__':
    main()
