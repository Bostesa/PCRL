"""New queued exact useful-head/checkpoint admission; numerical worker, zero fitting."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

CAMPAIGN = 'pcrl_compressed_income_safeguard_v1'
REQUIRED_SOURCES = {
    'RAW_JOINT_R03_E5_S0': (0, 'a1a0dfba5a9036dffa3953f57a79c9e91d701b36ba490070ba77fdb33d984a4c', 5, 'RAW_JOINT'),
    'RAW_JOINT_R03_E5_S1': (1, 'b10a389a6c73943765afc40b3014c9580c42c6c73700cd7a62291b7810cb7bdf', 5, 'RAW_JOINT'),
    'U0_ANCHOR_S0': (0, 'b94095b78255ff523cc4d93aa0973dc219abd2bbdcd57638f0be28978ec13a25', 0, 'U_NEW'),
    'U0_ANCHOR_S1': (1, '9dc8f71ef4be18f1e39dbe170b4038db07e7f24e0c6afed4985dfe47e0e7f425', 0, 'U_NEW'),
}


def validate_source_metadata(manifest, roster):
    """Data-free whitelist before any model or NPZ bytes are touched."""
    from .admission import admitted_path, ROLES
    if manifest.get('schema') != 'OB1-role-safe-private-source-v1':
        raise PermissionError('inherited source schema required')
    if manifest.get('provenance', {}).get('compressed_income_safeguard', {}).get('campaign') != CAMPAIGN:
        raise PermissionError('new campaign source metadata binding required')
    expected_fields = {
        'OSF_DEFENSE_FIT': {'features', 'income', 'occupation', 'sex', 'references_s0', 'references_s1'},
        'ATTACK_FIT_NEW': {'features', 'sex', 'references_s0', 'references_s1'},
        'INNER_SELECTION': {'features', 'references_s0', 'references_s1'},
    }
    if set(manifest.get('roles', {})) != set(expected_fields):
        raise PermissionError('exact three inherited roles required')
    for role, fields in expected_fields.items():
        if set(manifest['roles'][role]) != fields:
            raise PermissionError('role fields differ from prefit whitelist: ' + role)
        for item in manifest['roles'][role].values():
            admitted_path(item['path'])
    if set(manifest.get('seeds', {})) != {'0', '1'}:
        raise PermissionError('both inherited source seeds required')
    if roster.get('schema') != 'OB1-checkpoint-metadata-inventory-v1' or roster.get('campaign') != CAMPAIGN:
        raise PermissionError('new campaign reduced roster required')
    expected_counts = {role: {'rows': count[0], 'groups': count[1]} for role, count in ROLES.items()}
    if roster.get('expected_role_counts') != expected_counts:
        raise PermissionError('inherited role count metadata differs')
    entries = roster.get('entries', [])
    if len(entries) != 4 or {e.get('unit_id') for e in entries} != set(REQUIRED_SOURCES):
        raise PermissionError('exact four required source instances must be present')
    for entry in entries:
        seed, digest, epoch, family = REQUIRED_SOURCES[entry['unit_id']]
        if (entry.get('seed'), entry.get('checkpoint_sha256'), entry.get('epoch'), entry.get('family')) != (seed, digest, epoch, family):
            raise PermissionError('required source checkpoint identity differs')
        if entry.get('checkpoint_schema') != 'TCR1-trainable-own-head-checkpoint-v1':
            raise PermissionError('inherited checkpoint schema required')
        if not entry.get('exists'):
            raise PermissionError('INCOMPLETE_REQUIRED_ARTIFACT: ' + entry['unit_id'])
        alias = f'U0_S{seed}' if epoch == 0 else entry['unit_id']
        if entry.get('canonical_model_alias') != alias or (epoch == 5 and entry.get('strength') != .3):
            raise PermissionError('required source alias or strength differs')
        admitted_path(entry['checkpoint'])
    return entries


def load_useful_model(manifest, seed):
    """Restore the exact useful source through this campaign's queued pins.

    UsefulModel and fold_estimator retain the inherited tensor/head arithmetic;
    its old module's load_useful_model is deliberately not called because that
    function binds its pin loader to the predecessor queue.
    """
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')
    import joblib, numpy as np, torch
    from .source import pinned
    from task_conditioned_representation.useful_service import UsefulModel, fold_estimator
    row = manifest['seeds'][str(seed)]; pins = row['pins']
    for item in pins.values(): pinned(item)
    complete = json.loads(pinned(pins['complete']).read_text())
    for filename, key in (('model.pt', 'model'), ('head_0.joblib', 'head0'), ('head_1.joblib', 'head1'), ('record.json', 'record')):
        if complete['files'].get(filename) != pins[key]['sha256']:
            raise PermissionError('source COMPLETE binding differs')
    record = json.loads(pinned(pins['record']).read_text())
    if int(record['seed']) != seed: raise PermissionError('source seed mismatch')
    bank = json.loads(pinned(row['public_bank']).read_text())
    model = UsefulModel(bank['feature_names'])
    state = torch.load(pinned(pins['model']), map_location='cpu', weights_only=True)
    own = model.state_dict()
    for key in own:
        if key.startswith('enc.'): own[key] = state[key]
    model.load_state_dict(own, strict=True)
    if any(not torch.equal(value, state[key]) for key, value in model.state_dict().items() if key.startswith('enc.')):
        raise PermissionError('source encoder tensor parity failed')
    heads = []
    for i, key in enumerate(('head0', 'head1')):
        w, b, meta = fold_estimator(joblib.load(pinned(pins[key])))
        with torch.no_grad():
            model.head[i].weight.copy_(torch.from_numpy(w)); model.head[i].bias.copy_(torch.from_numpy(b))
        heads.append(meta)
    model.source_bindings = {'seed': seed, 'feature_schema_sha256': bank['feature_schema_sha256'],
                             'files': {key: value['sha256'] for key, value in pins.items()}, 'heads': heads}
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise PermissionError('useful source tensors nonfinite')
    model.eval(); return model


def arr_sha(a):
    import numpy as np
    z=np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode()+str(z.shape).encode()+z.tobytes()).hexdigest()


def metrics(prob,y):
    import numpy as np
    p=np.asarray(prob,np.float64);n=len(y);k=p.shape[1];d=p.argmax(1)
    target=np.eye(k)[y];rec=[float((d[y==c]==c).mean()) if (y==c).any() else 0.0 for c in range(k)]
    return {'accuracy':float((d==y).mean()),'nll':float(-np.log(np.maximum(p[np.arange(n),y],1e-12)).mean()),'brier':float(((p-target)**2).sum(1).mean()),'balanced_accuracy':float(np.mean(rec)),'per_class_recall':rec,'class_counts':np.bincount(y,minlength=k).tolist(),'mean_confidence':float(p.max(1).mean())}


def strata(fit,mask):
    import numpy as np
    u=fit['unit'][mask];s=fit['sex'][mask];y1=fit['income'][mask];y2=fit['occupation'][mask]
    # Labels of an exact-record duplicate group must agree; support counts unique people.
    for g in np.unique(u):
        ix=u==g
        if len(np.unique(np.c_[s[ix],y1[ix],y2[ix]],axis=0))!=1:raise AssertionError('duplicate-group labels differ')
    def one(a,keys):
        records=[]
        for key in keys:
            hit=np.all(a==np.asarray(key)[None,:],axis=1);count=[len(np.unique(u[hit&(s==v)])) for v in (0,1)]
            records.append({'labels':list(key),'rows':int(hit.sum()),'mass':float(hit.mean()),'distinct_groups_by_sex':count,'supported':min(count)>=32})
        return {'strata':records,'covered_mass':sum(r['mass'] for r in records if r['supported']),'uncovered_mass':sum(r['mass'] for r in records if not r['supported'])}
    return {'income':one(y1[:,None],[(i,) for i in range(2)]),'occupation':one(y2[:,None],[(i,) for i in range(6)]),'pair':one(np.c_[y1,y2],[(i,j) for i in range(2) for j in range(6)])}


def run(manifest_path,manifest_sha256,output,coverage_output,private_output, roster_path=None, roster_sha256=None):
    # Import numerical packages only after scheduler has independently authenticated this child.
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')
    from .source import load_manifest,load_role,pinned,ROLES,write_json,sha_file
    from .admission import read_metadata,sha_metadata
    m=load_manifest(manifest_path,manifest_sha256)
    if not roster_path or not roster_sha256 or sha_metadata(roster_path)!=roster_sha256:
        raise PermissionError('exact reduced source roster and metadata hash required')
    roster=read_metadata(roster_path)
    if roster.get('source_paths_sha256')!=manifest_sha256:
        raise PermissionError('source roster/manifest binding differs')
    entries=validate_source_metadata(m,roster)
    import joblib,numpy as np,torch
    torch.set_num_threads(1)
    reports=[];pairings={};models={};source_models={};att_split_audit=None
    # Guard behavior is exercised before any request can reach filesystem hashing.
    for role,field in [('INNER_SELECTION','sex'),('INNER_SELECTION','income'),('INNER_SELECTION','occupation'),('ATTACK_FIT_NEW','income')]:
        try:load_role(m,0,role,(field,))
        except PermissionError:pass
        else:raise AssertionError('label guard admitted barred prefit field')
    for seed in (0,1):
        model=load_useful_model(m,seed);models[str(seed)]=model.source_bindings;source_models[seed]=model
        calpins=m['seeds'][str(seed)]['ucal'];cal=json.loads(pinned(calpins['record']).read_text());complete=json.loads(pinned(calpins['complete']).read_text())
        if complete['files']['record.json']!=calpins['record']['sha256'] or cal['schema']!='hcal-calU-v1' or cal['seed']!=seed:raise AssertionError('Ucal provenance differs')
        heads=[joblib.load(pinned(m['seeds'][str(seed)]['pins'][k])) for k in ('head0','head1')]
        for role in ROLES:
            fields={'features','references'}|({'income','occupation','sex'} if role=='OSF_DEFENSE_FIT' else {'sex'} if role=='ATTACK_FIT_NEW' else set())
            data=load_role(m,seed,role,fields)
            if role=='ATTACK_FIT_NEW' and seed==0:
                from task_conditioned_representation.attacks import attack_split
                train,val=attack_split(data['unit'])
                if np.intersect1d(data['unit'][train],data['unit'][val]).size or any(np.unique(data['sex'][ix]).size!=2 for ix in (train,val)):
                    raise PermissionError('inherited ATT training/validation pairing or SEX support differs')
                for group in np.unique(data['unit']):
                    if np.unique(data['sex'][data['unit']==group]).size!=1:
                        raise PermissionError('ATT duplicate-group SEX labels differ')
                att_split_audit={'train_rows':len(train),'validation_rows':len(val),'train_groups':len(np.unique(data['unit'][train])),
                                 'validation_groups':len(np.unique(data['unit'][val])),'group_overlap':0,
                                 'training_indices_sha256':arr_sha(train),'validation_indices_sha256':arr_sha(val),
                                 'both_SEX_classes_in_each_split':True,'selection_and_orientation_role':'ATT validation only'}
            if role in pairings:
                if any(not np.array_equal(data[k],pairings[role][k]) for k in ('row_id','unit')):raise AssertionError('source seeds not paired')
            else:pairings[role]={k:data[k] for k in ('row_id','unit')}
            if tuple(map(str,data['feature_names']))!=model.names:raise AssertionError('source feature schema differs')
            wires=model.route_raw(data['X'],data['feature_names']);tasks=[]
            for i in range(2):
                z=wires[i]['representation'];p=wires[i]['probability'];reference=heads[i].predict_proba(z.astype(np.float64));stored=data['references'][f'U0_{i+1}']
                gap=float(np.max(np.abs(p-reference)));storedgap=float(np.max(np.abs(reference-stored)))
                if gap>2e-12 or storedgap>2e-12 or not np.array_equal(p.argmax(1),reference.argmax(1)):raise AssertionError(f'folded useful head probability parity failed {seed} {role} {i}: {gap}/{storedgap}')
                if not np.array_equal(p.argmax(1),data['references'][f'd_{i+1}']):raise AssertionError('stored decision mismatch')
                alpha=float(cal['tables'][f'H-GLOBAL-TEMP|r{i+1}']['alpha'])
                logp=np.log(np.maximum(reference,1e-12));logp-=np.log(np.maximum(reference,1e-12).sum(1))[:,None];v=alpha*logp;v=np.exp(v-v.max(1,keepdims=True));q=v/v.sum(1,keepdims=True)
                ucgap=float(np.max(np.abs(q-data['references'][f'Ucal_{i+1}'])))
                if ucgap>2e-12:raise AssertionError('Ucal stored probability differs')
                item={'task':('income','occupation')[i],'folded_vs_deployed_max_abs':gap,'deployed_vs_stored_max_abs':storedgap,'Ucal_vs_stored_max_abs':ucgap,'Ucal_alpha':alpha,'decision_parity':True,'representation_sha256':arr_sha(z),'wire_dimension':wires[i]['wire'].shape[1]}
                if role=='OSF_DEFENSE_FIT':
                    y=data[('income','occupation')[i]];item['FIT_metrics']={'deployed_U0':metrics(reference,y),'folded_U0':metrics(p,y),'stored_Ucal':metrics(data['references'][f'Ucal_{i+1}'],y)}
                tasks.append(item)
            reports.append({'seed':seed,'role':role,'rows':len(data['X']),'groups':len(np.unique(data['unit'])),'row_id_sha256':arr_sha(data['row_id']),'group_sha256':arr_sha(data['unit']),'tasks':tasks})
        if seed==0:fit=load_role(m,0,'OSF_DEFENSE_FIT',{'features','income','occupation','sex'})
    overlap=[]
    for a,b in [('OSF_DEFENSE_FIT','ATTACK_FIT_NEW'),('OSF_DEFENSE_FIT','INNER_SELECTION'),('ATTACK_FIT_NEW','INNER_SELECTION')]:
        rr=len(np.intersect1d(pairings[a]['row_id'],pairings[b]['row_id']));gg=len(np.intersect1d(pairings[a]['unit'],pairings[b]['unit']))
        if rr or gg:raise AssertionError('roles overlap')
        overlap.append({'a':a,'b':b,'rows':rr,'groups':gg})
    folds={str(i):{'rows':int((fit['fold']==i).sum()),'groups':len(np.unique(fit['unit'][fit['fold']==i]))} for i in range(3)}
    if folds['0']['rows']+folds['1']['rows']!=10223 or folds['2']['rows']!=5211:
        raise PermissionError('inherited FIT update/selection fold counts differ')
    coverage={'schema':'OB1-stratum-coverage-v1','minimum_distinct_groups_per_sex':32,'whole_FIT':strata(fit,np.ones(len(fit['X']),bool)),'FIT_train_folds01':strata(fit,fit['fold']!=2),'FIT_selection_fold2':strata(fit,fit['fold']==2),'fold_counts':folds,'FIT_fold_sha256':arr_sha(fit['fold']),'no_INNER_labels_opened':True}
    private={'schema':'OB1-useful-source-bindings-v1','campaign':CAMPAIGN,'models':models,'feature_names':list(model.names),'source_manifest_sha256':manifest_sha256,'parameters':model.n_params()}
    report={'schema':'CIS1-source-admission-v1','campaign':CAMPAIGN,'status':'PASS','source_manifest_sha256':manifest_sha256,'source_roles':reports,'role_overlap':overlap,'fold_counts':folds,'parameters':model.n_params(),'class_orders':model.class_orders,'head_metadata':{s:models[s]['heads'] for s in models},'float_convention':'source encoders float32; folded heads float64; inherited reference audit whole-role; canonical new wires fixed batches of 1024','probability_tolerance':2e-12,'INNER_task_labels_opened':False,'INNER_SEX_opened_or_hashed':False,'new_fits':0,'guard_fixtures':'PASS'}
    from task_conditioned_representation.models import load_checkpoint
    from .wires import bind_geometry,prepare_geometry,forward_role,FrozenIncomeGuard,WIRE_CONTRACT,ROLE_FORWARD_BATCH_SIZE
    from .geometry import audit_geometry
    admitted=[];alias_checks=[];geometry_bindings={};geometry_audits=[]
    for entry in entries:
        path=pinned({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']})
        actual,payload=load_checkpoint(path,entry['checkpoint_sha256'])
        seed=entry['seed'];source=source_models[seed]
        if (payload.get('seed'),payload.get('epoch'),payload.get('family'))!=(seed,entry['epoch'],entry['family']):
            raise PermissionError('checkpoint payload source identity differs')
        if entry['epoch']==5 and payload.get('strength')!=.3:
            raise PermissionError('RAW starting strength differs')
        if actual.names!=source.names or actual.class_orders!=source.class_orders:
            raise AssertionError('checkpoint schema or class order differs')
        if any(not torch.isfinite(v).all() for v in actual.state_dict().values()):raise AssertionError('checkpoint nonfinite')
        row={'unit_id':entry['unit_id'],'checkpoint_sha256':entry['checkpoint_sha256'],
             'checkpoint_schema':payload['schema'],'status':'PASS','seed':seed,
             'canonical_model_alias':entry['canonical_model_alias'],'epoch':entry['epoch'],'family':entry['family']}
        if entry['canonical_model_alias'].startswith('U0_S'):
            equal=all(torch.equal(v,source.state_dict()[k]) for k,v in actual.state_dict().items())
            if not equal:raise AssertionError('zero-update/source tensor alias differs')
            row['exact_source_tensor_alias']=True
            alias_checks.append({'unit_id':entry['unit_id'],'source_seed':seed,'exact_model_tensor_parity':True})
        binding=bind_geometry(actual,source_seed=seed,checkpoint_sha256=entry['checkpoint_sha256'])
        prepared=prepare_geometry(binding,source_seed=seed,checkpoint_sha256=entry['checkpoint_sha256'])
        if binding['heads'][0]['rank']!=1:
            raise PermissionError('admitted income rank differs from corrected rank-one wire contract')
        guard=FrozenIncomeGuard(actual,prepared)
        geometry_bindings[entry['unit_id']]=binding
        row['geometry_binding_sha256']=binding['binding_sha256']
        is_u0=entry['canonical_model_alias'].startswith('U0_S')
        for role in ROLES:
            data=load_role(m,seed,role,{'features'})
            full=forward_role(actual,data['X'],data['feature_names'],prepared,variant='U0_FULL' if is_u0 else 'RAW_FULL')
            projected=forward_role(actual,data['X'],data['feature_names'],prepared,variant='U0_PROJ' if is_u0 else 'RAW_PROJ')
            hybrid=None if is_u0 else forward_role(actual,data['X'],data['feature_names'],prepared,variant='RAW_HYBRID')
            audit_rows=[]
            for i in range(2):
                g={key:np.asarray(value) if key in {'weight','bias','class_order','A','c','B','P','singular_values'} else value for key,value in binding['heads'][i].items()}
                g['B']=g['B'].reshape(g['rank'],g['source_dimension'])
                audit=audit_geometry(full[i]['representation'],g)
                # Audit the actual Torch recipient wire as well as geometry arithmetic.
                logit_gap=float(np.max(np.abs(full[i]['centered_logits']-projected[i]['centered_logits'])))
                probability_gap=float(np.max(np.abs(full[i]['probability']-projected[i]['probability'])))
                decision_parity=bool(np.array_equal(full[i]['decision'],projected[i]['decision']))
                literal_pass=bool(np.allclose(full[i]['centered_logits'],projected[i]['centered_logits'],atol=g['rule']['logit_atol'],rtol=g['rule']['logit_rtol']) and np.allclose(full[i]['probability'],projected[i]['probability'],atol=g['rule']['probability_atol'],rtol=g['rule']['probability_rtol']) and decision_parity)
                if not audit['passed'] or not literal_pass:raise PermissionError('factorized own-head projection parity failed')
                width=(21,29)[i]
                if full[i]['wire'].shape[1]!=width or projected[i]['wire'].shape[1]!=width:
                    raise PermissionError('actual recipient width differs')
                if hybrid is not None and not np.array_equal(hybrid[i]['wire'],(projected if i==0 else full)[i]['wire']):
                    raise PermissionError('RAW hybrid income/occupation alias differs from literal canonical wire')
                audit_rows.append({'recipient':('income','occupation')[i],'geometry':audit,'actual_wire_centered_logits_max_abs':logit_gap,'actual_wire_probability_max_abs':probability_gap,'actual_wire_decision_parity':decision_parity,'wire_width':width,'full_wire_sha256':arr_sha(full[i]['wire']),'projected_wire_sha256':arr_sha(projected[i]['wire']),'hybrid_wire_sha256':arr_sha(hybrid[i]['wire']) if hybrid is not None else None})
            guard.verify()
            geometry_audits.append({'unit_id':entry['unit_id'],'seed':seed,'role':role,'rows':len(data['X']),'geometry_binding_sha256':binding['binding_sha256'],'tasks':audit_rows,'INNER_labels_loaded':False})
        admitted.append(row)
    report.update(checkpoint_roster_sha256=roster_sha256,checkpoint_packages=admitted,
                  exact_source_alias_checks=alias_checks,available_checkpoint_packages=4,missing_checkpoint_packages=0,
                  ATT_group_split=att_split_audit,
                  required_source_instances=list(REQUIRED_SOURCES),geometry_bindings=geometry_bindings,
                  geometry_audits=geometry_audits,canonical_wire_contract=WIRE_CONTRACT,
                  canonical_role_forward_batch_size=ROLE_FORWARD_BATCH_SIZE,
                  income_geometry_binding_by_seed={str(e['seed']):geometry_bindings[e['unit_id']]['binding_sha256'] for e in entries if e['epoch']==5})
    private.update(geometry_bindings=geometry_bindings,checkpoint_packages=admitted)
    write_json(output,report);write_json(coverage_output,coverage);write_json(private_output,private,True)
    return {'status':'PASS','output_sha256':sha_file(output),'coverage_sha256':sha_file(coverage_output)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--coverage-output',required=True);p.add_argument('--private-output',required=True);p.add_argument('--roster',required=True);p.add_argument('--roster-sha256',required=True)
    a=p.parse_args();print(json.dumps(run(a.manifest,a.manifest_sha256,a.output,a.coverage_output,a.private_output,a.roster,a.roster_sha256)))
