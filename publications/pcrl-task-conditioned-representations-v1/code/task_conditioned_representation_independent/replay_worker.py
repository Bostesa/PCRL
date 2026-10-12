"""Independent actual checkpoint/reader/utility/AUC/bootstrap replay.

Spec records list selected private checkpoints and frozen reader directories.
Only the queue guard is shared with candidate execution; every scientific
transformation, metric, nomination replay and interval is independently coded.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    spec=json.loads(Path(a.spec).read_text())
    def checked(pin):
        assert_active_from_env(stage='verification')
        if digest(pin['path'])!=pin['sha256']:raise ValueError('replay source hash differs: '+Path(pin['path']).name)
        return Path(pin['path'])
    # Explicit all-family and all-reader freezes precede either label opening.
    nomination=json.loads(checked(spec['nomination_lock']).read_text())
    reader_lock=json.loads(checked(spec['reader_lock']).read_text())
    if nomination.get('status')!='FROZEN' or not nomination.get('nominees'):
        raise PermissionError('complete nomination freeze required before INNER tasks')
    if not spec.get('all_reader_banks_frozen') or reader_lock.get('status') not in ('FROZEN','FROZEN_COMPLETE','FROZEN_ALL','ALL_READERS_FROZEN'):
        raise PermissionError('all-reader freeze required before INNER SEX')
    import numpy as np
    import torch
    from .science import own_release,wire,metrics,task_rows,auc
    from .checkpoint_io import read as read_checkpoint
    from .readers import READERS,features,predict,load_saved_reader,split,shuffled_groups
    from .endpoint_math import scores_envelope,bootstrap_counts,interval,clause
    torch.set_num_threads(1)
    source=json.loads(checked(spec['source_manifest']).read_text())
    role=source['roles']['INNER_SELECTION']
    def npz(pin):
        path=checked(pin)
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    x=npz(role['features']);ys=[]
    for name in ('income','occupation'):
        y=npz(role[name])
        if not np.array_equal(y['row_id'],x['row_id']) or not np.array_equal(y['unit'],x['unit']):raise ValueError('task person alignment differs')
        ys.append(y[name])
    sexdata=npz(spec['inner_sex'])
    if not np.array_equal(sexdata['row_id'],x['row_id']) or not np.array_equal(sexdata['unit'],x['unit']):raise ValueError('SEX person alignment differs')
    sex=sexdata['sex'];groups=x['unit'];ng=len(np.unique(groups))
    if len(groups)!=2235 or ng!=2234:raise ValueError('reporting role count differs')
    refs={seed:npz(role[f'references_s{seed}']) for seed in (0,1)}
    for ref in refs.values():
        if not np.array_equal(ref['row_id'],x['row_id']) or not np.array_equal(ref['unit'],groups):raise ValueError('reference person alignment differs')
    banks=spec['banks']
    if len(banks)!=20 or len({(r['family'],r['seed']) for r in banks})!=20:raise ValueError('complete ten-family/two-seed replay required')
    families=sorted({r['family'] for r in banks})
    all_points={};all_boots={};rows={};nominee_rows={};reports=[];restore=[];gate={};actual_gates={}
    started=time.process_time()
    for entry in banks:
        assert_active_from_env(stage='verification')
        family,seed=entry['family'],entry['seed'];key=(family,seed)
        checkpoint=checked({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']})
        payload=read_checkpoint(checkpoint)
        if payload['schema']!='TCR1-trainable-own-head-checkpoint-v1' or list(payload['feature_names'])!=list(map(str,x['feature_names'])) or payload['class_orders']!=[[0,1],[0,1,2,3,4,5]]:
            raise ValueError('checkpoint schema/class order mismatch')
        released=own_release(payload['model'],x['X'])
        root=Path(entry['directory']);freeze=checked({'path':str(root/'FROZEN_COMPLETE.json'),'sha256':entry['receipt_sha256']})
        frozen=json.loads(freeze.read_text())
        if frozen['status']!='FROZEN_COMPLETE' or frozen['source_seed']!=seed or frozen['INNER_SEX_opened']:raise ValueError('reader bank not frozen prospectively')
        saved=npz({'path':str(root/'INNER_RELEASE.npz'),'sha256':frozen['inner_release_sha256']})
        att=npz({'path':str(root/'ATT_RELEASE.npz'),'sha256':frozen['att_release_sha256']})
        scores=npz({'path':str(root/'INNER_SCORES.npz'),'sha256':frozen['score_sha256']})
        if not np.array_equal(saved['row_id'],x['row_id']) or not np.array_equal(saved['unit'],groups):raise ValueError('saved release person order mismatch')
        errors=[float(np.max(abs(wire(released,r)-saved[f'r{r}']))) for r in (1,2)]
        if max(errors)>1e-10:raise ValueError('independent actual model release mismatch')
        train,val=split(att['unit'])
        if not np.array_equal(train,att['train']) or not np.array_equal(val,att['validation']):raise ValueError('reader split mismatch')
        reader_errors=[]
        for view in ('r1','r2','pair'):
            for name in READERS:
                filename=f'MODEL_{view}__{name}.joblib'
                path=checked({'path':str(root/filename),'sha256':frozen['model_files'][filename]})
                record=load_saved_reader(path)
                # Independently restore ATT-trained standardization and orientation.
                feature=features(att['r1'],att['r2'],view,name)
                sc=record['scaler']
                raw_std=feature[train].std(0)
                declared_scale=np.where(raw_std>0,raw_std,1.0)
                if max(float(np.max(abs(sc.mean_-feature[train].mean(0)))),float(np.max(abs(sc.scale_-declared_scale))))>1e-10:
                    raise ValueError('reader scaler not the declared ATT-training fit')
                vs=predict(record,feature[val]);sign=1 if auc(att['sex'][val],vs)>=.5 else -1
                if sign!=frozen['validation'][view][name]['orientation']:raise ValueError('reader orientation mismatch')
                p=predict(record,features(saved['r1'],saved['r2'],view,name))
                if sign==-1:p=1-p
                err=float(np.max(abs(p-scores[f'{view}__{name}'])))
                if err>1e-10:raise ValueError('independent saved-reader predictions mismatch')
                reader_errors.append(err)
        for view,other in [('r1','r2'),('r2','r1')]:
            for name in READERS:
                if not np.array_equal(scores[f'pair__IGNORE_{other}__{name}'],scores[f'{view}__{name}']):raise ValueError('ignore-other alias differs')
        controls={}
        shuffled=shuffled_groups(sex,groups,2026101207)
        for view in ('r1','r2','pair'):
            p=scores[f'{view}__SHUFFLED'];sauc=auc(shuffled,p)
            planted=np.where(sex==1,scores[f'{view}__PLANTED_1'],scores[f'{view}__PLANTED_0'])
            controls[view]={'null_auc':auc(sex,np.full(len(sex),.5)),'shuffled_auc':max(sauc,1-sauc),'planted_auc':auc(sex,planted)}
            controls[view]['pass']=controls[view]['null_auc']==.5 and controls[view]['shuffled_auc']<=.6 and controls[view]['planted_auc']>=.99
        points,boots=scores_envelope(scores,sex,groups);all_points[key]=points;all_boots[key]=boots
        tm=[metrics(ys[r],released[f'p{r+1}']) for r in (0,1)]
        rows[key]=[task_rows(ys[r],released[f'p{r+1}']) for r in (0,1)]
        nominal_path=checked({'path':entry['nominee_checkpoint'],'sha256':entry['nominee_checkpoint_sha256']})
        nominal_payload=read_checkpoint(nominal_path)
        if nominal_payload['seed']!=seed:raise ValueError('nominee checkpoint seed differs')
        nominal_release=own_release(nominal_payload['model'],x['X'])
        nominal_tm=[metrics(ys[r],nominal_release[f'p{r+1}']) for r in (0,1)]
        nominee_rows[key]=[task_rows(ys[r],nominal_release[f'p{r+1}']) for r in (0,1)]
        clauses=[];actual_clauses=[]
        for r in (0,1):
            for refname in ('U0','Ucal'):
                baseline=metrics(ys[r],refs[seed][f'{refname}_{r+1}'])
                for metric,cap in [('accuracy',.010),('nll',.010),('brier',.005)]:
                    deviation=baseline[metric]-nominal_tm[r][metric] if metric=='accuracy' else nominal_tm[r][metric]-baseline[metric]
                    clauses.append({'task':r,'reference':refname,'metric':metric,'point':deviation,'cap':cap,'pass':deviation<=cap})
                    actual_deviation=baseline[metric]-tm[r][metric] if metric=='accuracy' else tm[r][metric]-baseline[metric]
                    actual_clauses.append({'task':r,'reference':refname,'metric':metric,'point':actual_deviation,'cap':cap,'pass':actual_deviation<=cap})
        gate[key]=all(c['pass'] for c in clauses)
        actual_gate=all(c['pass'] for c in actual_clauses)
        actual_gates[key]=actual_gate
        restore.append({'family':family,'seed':seed,'checkpoint_sha256':entry['checkpoint_sha256'],'release_max_abs_errors':errors,'max_reader_prediction_error':max(reader_errors),'readers_replayed':27})
        reports.append({'family':family,'seed':seed,'task_metrics':tm,'nominee_task_metrics':nominal_tm,
                        'nominee_point_eligible_per_seed':gate[key],'actual_point_eligible_per_seed':actual_gate,'diagnostic_only':bool(entry['diagnostic_only']),
                        'nominee_point_clauses':clauses,'actual_model_point_clauses':actual_clauses,'controls':controls,
                        'recovery':{contract:{view:points[(contract,view)] for view in ('r1','r2','pair')} for contract in ('full','score','representation')}})
    for entry in banks:
        family=entry['family'];eligible=all(gate[(family,s)] for s in (0,1));actual_eligible=all(actual_gates[(family,s)] for s in (0,1))
        if eligible!=entry['nominee_point_eligible'] or actual_eligible!=entry['point_eligible_of_actualmodel'] or bool(entry['diagnostic_only'])==eligible:
            raise ValueError('common two-seed nominee/fallback utility classification differs')
        if eligible and entry['checkpoint_sha256']!=entry['nominee_checkpoint_sha256']:raise ValueError('qualified nominee replaced after task gate')
    primary={};secondaryutility={};scoreprivacy={};absolute={}
    for family in families:
        eligible=all(gate[(family,s)] for s in (0,1))
        for contract in ('full','score','representation'):
            for view in ('r1','r2','pair'):
                point=sum(all_points[(family,s)][(contract,view)] for s in (0,1))/2
                b=sum(all_boots[(family,s)][(contract,view)] for s in (0,1))/2
                absolute[f'{family}__{contract}__{view}']=dict(interval(point,b,1,ng),point_eligible=eligible)
        for seed in (0,1):
            for r in (0,1):
                for refname in ('U0','Ucal'):
                    base=task_rows(ys[r],refs[seed][f'{refname}_{r+1}'])
                    candidate=rows[(family,seed)][r]
                    diff=candidate-base;diff[:,0]*=-1
                    reps=np.array([np.average(diff,axis=0,weights=w) for w in bootstrap_counts(groups)])
                    points=diff.mean(axis=0)
                    for j,(metric,cap) in enumerate([('accuracy',.010),('nll',.010),('brier',.005)]):
                        id=f'{family}__s{seed}__task{r}__{refname}__{metric}'
                        secondaryutility[id]=clause(points[j],reps[:,j],240,ng,cap,'upper')
                        if family=='JOINT_CONDITIONAL':
                            nominal_diff=nominee_rows[(family,seed)][r]-base;nominal_diff[:,0]*=-1
                            nominal_reps=np.array([np.average(nominal_diff,axis=0,weights=w) for w in bootstrap_counts(groups)])
                            primary[id]=clause(nominal_diff.mean(axis=0)[j],nominal_reps[:,j],51,ng,cap,'upper')
    for comparator in families:
        if comparator=='JOINT_CONDITIONAL':continue
        eligible=all(gate[(comparator,s)] for s in (0,1))
        for view in ('r1','r2','pair'):
            for contract,dest,k in [('full',primary,51),('score',scoreprivacy,33)]:
                point=sum(all_points[(comparator,s)][(contract,view)]-all_points[('JOINT_CONDITIONAL',s)][(contract,view)] for s in (0,1))/2
                reps=sum(all_boots[(comparator,s)][(contract,view)]-all_boots[('JOINT_CONDITIONAL',s)][(contract,view)] for s in (0,1))/2
                if view!='pair':point,reps=-point,-reps
                dest[f'{comparator}__{view}']=dict(clause(point,reps,k,ng,.020 if view=='pair' else .010,'lower' if view=='pair' else 'upper'),comparator_eligible=eligible)
    if len(primary)!=51 or len(secondaryutility)!=240 or len(scoreprivacy)!=27:raise ValueError('registered endpoint slots lost')
    # Legacy score slots remain explicit dependencies unless separately replayed.
    for legacy in ('P0','S12'):
        for view in ('r1','r2','pair'):scoreprivacy[f'{legacy}__{view}']={'status':'PENDING_LEGACY_DEPENDENCY','K':33}
    if any(name.startswith('task_conditioned_representation.') and name!='task_conditioned_representation.scheduler' for name in sys.modules):
        raise ImportError('candidate scientific implementation imported during independent replay')
    out={'schema':'TCR1-R6-independent-final-replay-v1','status':'PASS_NEW_CONTINUOUS_SERVICE_REPLAY','guard':guard,
         'model_and_reader_restore':restore,'complete_view_recovery':reports,'absolute_recovery':absolute,
         'primary_endpoints':primary,'secondary_utility_endpoints':secondaryutility,'score_only_endpoints':scoreprivacy,
         'point_eligibility':{f'{family}__s{seed}':value for (family,seed),value in gate.items()},
         'source_seeds_are_paired_fitted_runs':True,'bootstrap_replicates':2048,'groups':ng,'cpu_s':time.process_time()-started,
         'reader_refit_scope':'separate four predeclared independent pair reader fits; this worker restores saved readers',
         'legacy_score_scope':'six registered legacy score slots retained pending independent legacy artifact replay'}
    p=Path(spec['output']);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)

if __name__=='__main__':main()
