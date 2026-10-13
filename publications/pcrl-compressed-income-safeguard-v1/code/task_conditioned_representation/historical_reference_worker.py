"""Queued, label-free restoration of the fixed historical RAW-J C* reference.

Never calls historical whole-unit verification or an all-role release loader.
No fitting or task/protected-label reads are performed here.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path


def run(spec_path):
    from .scheduler import assert_active_from_env
    active=assert_active_from_env(stage='admission')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={str(Path(x['path']).resolve()):x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def verify(path,expected):
        p=Path(path)
        if p.is_symlink() or registered.get(str(p.resolve()))!=expected:raise PermissionError('historical reference source not queue-admitted')
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise PermissionError('historical reference source bytes differ')
        return p
    cfg=json.loads(verify(spec_path,registered.get(str(Path(spec_path).resolve()))).read_text())
    if cfg['unit_id']!=active['unit_id'] or cfg['arm_id']!='H_RAW_CSTAR' or cfg['seed'] not in (0,1):raise PermissionError('unregistered historical reference unit')
    scope=json.loads(verify(cfg['scope_lock'],cfg['scope_lock_sha256']).read_text())
    if scope.get('status')!='FROZEN' or scope.get('arm_id')!='H_RAW_CSTAR' or scope.get('K_primary')!=51 or scope.get('primary_pool_changed') is not False:raise PermissionError('additive historical-reference scope lock absent')
    verify(cfg['science_lock'],cfg['science_lock_sha256'])
    mp=verify(cfg['manifest'],cfg['manifest_sha256']);manifest=json.loads(mp.read_text())
    def pins(v):
        if isinstance(v,dict):
            if 'path' in v and 'sha256' in v:verify(v['path'],v['sha256'])
            else:
                for x in v.values():pins(x)
        elif isinstance(v,list):
            for x in v:pins(x)
    pins(manifest['seeds'][str(cfg['seed'])])
    pins(manifest['historical_contract'])
    for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):pins(manifest['roles'][role]['features'])
    # Queue authentication precedes all numerical package imports.
    import joblib,numpy as np,torch
    torch.set_num_threads(1)
    from .source import load_manifest,load_role,pinned,write_json,sha_file,ROLES
    from .useful_service import load_useful_model
    from .models import load_checkpoint
    m=load_manifest(mp,cfg['manifest_sha256']);seed=cfg['seed'];model=load_useful_model(m,seed)
    p=m['seeds'][str(seed)]['pins'];source=torch.load(pinned(p['model']),map_location='cpu',weights_only=True)
    heads=[joblib.load(pinned(p[k])) for k in ('head0','head1')]
    if m['historical_contract']['feature_schema_sha256']!=model.source_bindings['feature_schema_sha256']:raise PermissionError('historical/current raw83 schema mismatch')
    root=Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():raise PermissionError('fresh private reference output required')
    root.mkdir(parents=True,mode=0o700)
    payload={'schema':'TCR1-trainable-own-head-checkpoint-v1','model':model.state_dict(),'feature_names':list(model.names),'class_orders':model.class_orders,'source_bindings':model.source_bindings,'family':'H_RAW_CSTAR','seed':seed,'strength':0.,'coefficient':0.,'epoch':0,'updates_per_recipient':[0,0],'recipe':{'kind':'fixed_historical_restoration','new_updates':0,'historical_training_epochs':40,'historical_beta':.3},'optimizer':None,'selection_history':[],'selection_active':None,'selection_family':None,'historical_reference_only':True,'additive_scope_lock_sha256':cfg['scope_lock_sha256'],'comparison_scope':'descriptive additional reference outside prelocked primary pool'}
    checkpoint=root/'epoch_0.pt';torch.save(payload,checkpoint);checkpoint.chmod(0o600);checkpoint_sha=sha_file(checkpoint)
    restored,restored_payload=load_checkpoint(checkpoint,checkpoint_sha);reports=[]
    for role in ROLES:
        data=load_role(m,seed,role,{'features'});raw=torch.from_numpy(np.ascontiguousarray(data['X']));wire=model.route_raw(data['X'],data['feature_names']);back=restored.route_raw(data['X'],data['feature_names']);tasks=[]
        for i in range(2):
            # Independent direct source-weight computation avoids the historical data loader.
            h=raw
            with torch.no_grad():
                for layer in (0,2,4):
                    h=torch.nn.functional.linear(h,source[f'enc.{i}.{layer}.weight'],source[f'enc.{i}.{layer}.bias'])
                    if layer!=4:h=torch.relu(h)
            z=h.double().numpy();refp=heads[i].predict_proba(z);margin=np.asarray(heads[i].decision_function(z),np.float64)
            if margin.ndim==1:margin=np.stack((np.zeros_like(margin),margin),1)
            refl=margin-margin.mean(1,keepdims=True);refd=refp.argmax(1)
            zgap=float(np.max(np.abs(wire[i]['representation']-z)));pgap=float(np.max(np.abs(wire[i]['probability']-refp)));lgap=float(np.max(np.abs(wire[i]['centered_logits']-refl)))
            if zgap!=0 or pgap>2e-12 or lgap>2e-12 or not np.array_equal(wire[i]['decision'],refd):raise AssertionError('historical actual service forward differs')
            if any(not np.array_equal(wire[i][k],back[i][k]) for k in ('representation','centered_logits','probability','decision','wire')):raise AssertionError('zero-update checkpoint restore differs')
            tasks.append({'task':('income','occupation')[i],'representation_max_abs_gap':zgap,'probability_max_abs_gap':pgap,'centered_logits_max_abs_gap':lgap,'decision_exact':True,'checkpoint_restore_exact':True,'wire_dimension':wire[i]['wire'].shape[1],'class_order':model.class_orders[i]})
        reports.append({'role':role,'rows':len(data['X']),'groups':len(np.unique(data['unit'])),'tasks':tasks})
    output={'schema':'TCR1-historical-reference-restoration-v1','status':'PASS','arm_id':'H_RAW_CSTAR','seed':seed,'unit_id':active['unit_id'],'manifest_sha256':cfg['manifest_sha256'],'scope_lock_sha256':cfg['scope_lock_sha256'],'checkpoint_sha256':checkpoint_sha,'source_bindings':model.source_bindings,'parameters':model.n_params(),'roles':reports,'historical_configuration':'RAW-J|b0.3 / epoch40','new_fit_updates':0,'task_or_SEX_labels_read':False,'all_role_or_old_assessment_or_HEAD_arrays_opened_or_hashed':False,'K_primary':51,'primary_pool_changed':False,'utility_or_privacy_outcomes_computed':False,'current_point_utility_and_fresh_reader_status':'PENDING_SEPARATE_GLOBALLY_FROZEN_EVALUATION'}
    write_json(root/'RESTORATION_RECEIPT.json',output,True);write_json(cfg['public_output'],output)
    return {'status':'PASS','checkpoint_sha256':checkpoint_sha,'public_receipt_sha256':sha_file(cfg['public_output'])}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(run(a.spec)))
