"""Frozen-score residual spectral report. No model fitting or outcome reselection.

Full report requires 42 checksum-valid completed units after the global 24-map
freeze. --allow-incomplete produces an explicitly partial preview, never a decision.
Only aggregate evidence is exported; cached individual predictions stay local.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.acs_coalition_strength_comparisons import (SOURCE_TASKS, UTILITY_TASKS,
    AUDIT_ROLES, source_check, evaluate_pair, finite)
from scripts.summarize_acs_pca16_init import utility_criteria

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_ROOT=Path('/Users/nathansamson/PCRL')
HIST=('H','E','A0','L025','L20','J')
SPECTRAL=tuple('spectral_'+a for a in ('S0','M025','M1','L025','L1','C025','C1','L2'))
CONDITIONS=HIST+SPECTRAL
WEIGHTS=('unweighted','person_weighted')
SCOPES=('standard_independent','expanded_independent','expanded_catchup',
        'kernel_standard_independent','kernel_expanded_independent','kernel_expanded_catchup')
MAIN_SCOPE='kernel_expanded_catchup'
SENSITIVE=tuple(v+'/'+t for v in ('A','B','AB') for t in ('SEX','RAC1P'))
FORBIDDEN=tuple(v+'/'+t for v,ts in AUDIT_ROLES.items() for t in ts)
P_VALUES=(0.,.25,.5,.75,1.)
WITHHOLD=('E','A0','L025','L20')
MAIN_PAIRS=(('spectral_C1','spectral_L1'),('spectral_C1','spectral_L2'),('spectral_C025','spectral_L025'))
PAIRS=tuple(dict.fromkeys(MAIN_PAIRS+tuple((c,r) for c in SPECTRAL for r in ('J','H','E','A0'))+tuple(('J',r) for r in ('H','E','A0','L025','L20'))))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csvout(path,rows):
    """Compact aggregate-only export; raw IDs/loss vectors never enter rows."""
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows)


def completion_status(out,allow_incomplete=False):
    gate=Path(out)/'RELEASE_MANIFEST.json'
    if not gate.exists() or not read(gate).get('all24_frozen'):
        raise ValueError('global 24-map freeze must exist before any report')
    missing=[];available=[]
    for seed,condition in itertools.product(range(3),CONDITIONS):
        unit=Path(out)/f'seed_{seed}'/condition
        required=[unit/'metrics.json',unit/'predictions.npz',unit/'complete.json']
        if not all(p.exists() for p in required):missing.append([seed,condition]);continue
        receipt=read(unit/'complete.json')
        if receipt.get('metric_sha256')!=sha(unit/'metrics.json') or receipt.get('prediction_sha256')!=sha(unit/'predictions.npz'):
            raise ValueError(f'completed unit checksum mismatch: {seed}/{condition}')
        available.append([seed,condition])
    complete=not missing
    if not complete and not allow_incomplete:raise ValueError(f'full 42-unit matrix required; {len(available)}/42 complete')
    return {'complete':complete,'completed_units':len(available),'expected_units':42,'available':available,'missing':missing,
            'release_manifest_sha256':sha(gate),'status':'complete_development_matrix' if complete else 'INCOMPLETE_PREVIEW_NO_DECISION'}


def score_key(split,weight):
    return split+('_person_weighted' if weight=='person_weighted' else '')


def point_from_records(rows,priors,split,weight,budget,scope):
    key=score_key(split,weight); utility={};gains={};losses={};selected={};coverage={}
    for row in rows:
        if row['role']=='utility' and 'utility' in row['selected_scopes']:
            target=row['target']
            if target in utility:raise ValueError('Duplicate selected utility candidate')
            utility[target]=row['scores'][key]['log_loss'];selected['utility/'+target]=row
        elif row['role']=='audit' and row['audit_budget']==budget and scope in row['selected_scopes']:
            endpoint=row['view']+'/'+row['target']
            if endpoint in gains:raise ValueError('Duplicate selected audit candidate')
            loss=row['scores'][key]['log_loss'];losses[endpoint]=loss
            prior=priors[row['target']][key]['log_loss'];gains[endpoint]=prior-loss
            coverage[endpoint]=row['scores'][key].get('coverage_complete',False)
            selected['audit/'+endpoint]=row
    complete=set(utility)==set(UTILITY_TASKS) and set(gains)==set(FORBIDDEN)
    return {'utility':utility,'gains':gains,'losses':losses,'selected':selected,'coverage':coverage,
            'assessable':complete and all(coverage.get(e,False) for e in SENSITIVE)}


def person_losses(labels,predictions):
    y=np.asarray(labels);valid=y>=0;p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or len(p)!=int(valid.sum()):raise ValueError('prediction rows must match valid-label rows')
    if not np.isfinite(p).all():raise ValueError('nonfinite predictions')
    yy=y[valid].astype(int)
    if np.any(yy>=p.shape[1]):raise ValueError('prediction class schema mismatch')
    clipped=np.clip(p,1e-12,1.);clipped/=clipped.sum(axis=1,keepdims=True)
    return -np.log(clipped[np.arange(len(yy)),yy]),valid


def branch_uniforms(raw_rows,seed,condition):
    return np.array([int.from_bytes(hashlib.sha256(f'spectral-withholding-v1|{seed}|{condition}|{int(row)}'.encode()).digest()[:8],'big')/2**64 for row in raw_rows])


def withholding_losses(h,aug,p,uniforms):
    h,aug,uniforms=map(np.asarray,(h,aug,uniforms))
    if h.shape!=aug.shape or h.shape!=uniforms.shape or not 0<=p<=1:raise ValueError('aligned losses/uniforms and valid probability required')
    return (1-p)*h+p*aug,np.where(uniforms<p,aug,h)


def directional(left,right,tasks,delta,leak_tolerance=1e-12):
    du={t:left['utility'].get(t,np.nan)-right['utility'].get(t,np.nan) for t in tasks}
    dg={e:left['gains'].get(e,np.nan)-right['gains'].get(e,np.nan) for e in SENSITIVE}
    valid=all(np.isfinite(v) for v in [*du.values(),*dg.values()])
    utility=valid and all(v<=delta+1e-12 for v in du.values())
    sensitive=valid and all(v<=leak_tolerance for v in dg.values())
    strict=valid and (any(v< -1e-12 for v in du.values()) or any(v< -1e-12 for v in dg.values()))
    return {'assessable':valid,'utility_no_worse':utility,'sensitive_no_worse':sensitive,
            'strict_improvement':strict,'dominates':utility and sensitive and strict,
            'utility_differences':du,'sensitive_differences':dg}


def _fixed_vector_gate(points,left,right):
    rows=[]
    for seed,weight in itertools.product(range(3),WEIGHTS):
        key=(seed,left,'test',weight,360,MAIN_SCOPE);other=(seed,right,'test',weight,360,MAIN_SCOPE)
        if key not in points or other not in points:return {'pass':False,'reason':'missing_point'}
        a,b=points[key],points[other];d=directional(a,b,UTILITY_TASKS,.001,.0005+1e-12)
        rows.append({'seed':seed,'weight':weight,'full_schema_assessable':a.get('assessable',False) and b.get('assessable',False),**d})
    vector=all(r['assessable'] and r['utility_no_worse'] and r['sensitive_no_worse'] for r in rows)
    residence=all(r['utility_differences']['same_residence']<-.001 for r in rows)
    roles=[e for e in SENSITIVE if all(r['sensitive_differences'][e]<-.001 for r in rows)]
    return {'pass':vector and (residence or bool(roles)),'vector_nonworsening':vector,
            'full_schema_assessable':all(r['full_schema_assessable'] for r in rows),
            'strict_residence_all_seed_weights':residence,'shared_strict_sensitive_roles':roles,'per_seed_weight':rows}


def nominate(points,parents,complete):
    if not complete:return {'decision_status':'incomplete_no_decision','nominee':None,'qualifying_arms':[]}
    checks={}
    for arm in SPECTRAL:
        source=[];residence=[]
        for seed,split,weight in itertools.product(range(3),('validation','test'),WEIGHTS):
            point=points.get((seed,arm,split,weight,360,MAIN_SCOPE),{})
            check=source_check(point,parents.get((seed,split,weight),{}))
            source.append({'seed':seed,'split':split,'weight':weight,'pass':check['pass'],'tasks':check['tasks']})
            if split=='test':
                h=points.get((seed,'H',split,weight,360,MAIN_SCOPE),{})
                gain=h.get('utility',{}).get('same_residence',np.nan)-point.get('utility',{}).get('same_residence',np.nan)
                residence.append({'seed':seed,'weight':weight,'gain':float(gain) if np.isfinite(gain) else None,'pass':bool(gain>=.01-1e-12)})
        versus_j=_fixed_vector_gate(points,arm,'J')
        passed=all(r['pass'] is True for r in source+residence) and versus_j['pass']
        checks[arm]={'qualifies':passed,'source_all_validation_and_development':all(r['pass'] is True for r in source),
                     'residence_at_least_01_all_seed_weights':all(r['pass'] for r in residence),
                     'source_details':source,'residence_details':residence,'versus_J':versus_j}
    eligible=sorted(arm for arm,d in checks.items() if d['qualifies'])
    coordination={right:_fixed_vector_gate(points,'spectral_C1',right) for right in ('spectral_L1','spectral_L2')}
    return {'decision_status':'complete_development_selection','nominee':eligible[0] if eligible else None,
            'qualifying_arms':eligible,'tie_break':'lexicographic stored arm ID, fixed prospectively','checks':checks,
            'coordination':coordination,'coordination_supported':all(v['pass'] for v in coordination.values()),
            'no_further_sweep':True,'not_independent_confirmation':True}


def summarize(rows,identity,value='value'):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in identity)].append(row)
    result=[]
    for key,group in groups.items():
        values=[r[value] for r in group if finite(r[value])]
        result.append({**dict(zip(identity,key)),'n_seeds':len({r['seed'] for r in group}),
                       'n_defined':len(values),'mean':statistics.mean(values) if values else None,
                       'sample_sd':statistics.stdev(values) if len(values)>1 else None,
                       'all_three_seeds':len({r['seed'] for r in group})==3})
    return result


def fit_class_support(metadata,class_index):
    support=metadata.get('fit_support')
    return support[class_index] if support is not None and class_index<len(support) else None


def load_evidence(out,status,original_root):
    historical=original_root/'results/redesign_20260909_acs_fixed_predictions_v1'
    rules=read(historical/'comparison_rules.json');points={};candidate=[];classes=[];flat=[];parents={};raw={};sources={}
    available={tuple(x) for x in status['available']}
    for seed in range(3):
        controls=original_root/f'results/redesign_20260908_acs_coalition_v1/seed_{seed}/controls/metrics.json'
        sources[str(controls)]=sha(controls)
        prior={r['target']:r['scores'] for r in read(controls)['raw_metrics'] if r['condition']=='prior'}
        for split,weight in itertools.product(('validation','test'),WEIGHTS):
            k=score_key(split,weight)
            parents[seed,split,weight]={t:r['scores'][k] for t,r in rules['original_parent_metric_identity'][str(seed)]['tasks'].items()}
        for condition in CONDITIONS:
            if (seed,condition) not in available:continue
            path=out/f'seed_{seed}'/condition/'metrics.json';records=read(path)['raw_metrics'];raw[seed,condition]=records;sources[str(path)]=sha(path)
            metadata_path=path.parent/'audits/audit_selection.json'
            audit_metadata=read(metadata_path)['candidates'];sources[str(metadata_path)]=sha(metadata_path)
            for row in records:
                for split,weight in itertools.product(('validation','test'),WEIGHTS):
                    score=row['scores'][score_key(split,weight)]
                    base={k:row.get(k) for k in ('seed','condition','role','view','target','audit_budget','candidate_id','selected_scopes')}
                    base.update(seed=seed,condition=condition,split=split,weight=weight)
                    meta={}
                    if row['role']=='audit':
                        meta=audit_metadata[str(row['audit_budget'])][row['view']+'/'+row['target']][row['candidate_id']]
                        base.update({k:meta.get(k) for k in ('family','candidate_origin','source_view','source_candidate_id','fit_rows','fit_support','fit_coverage_complete','space','reused_historical_fit','inherited_singleton')})
                    candidate.append({**base,**{k:v for k,v in score.items() if k!='per_class'},
                        'absolute_recovery':prior[row['target']][score_key(split,weight)]['log_loss']-score['log_loss'] if row['role']=='audit' else None})
                    classes.extend({**base,**r,'coverage_complete':score.get('coverage_complete'),
                        'fit_class_support':fit_class_support(meta,r['class_index'])} for r in score.get('per_class',[]))
            for split,weight,budget,scope in itertools.product(('validation','test'),WEIGHTS,(120,360),SCOPES):
                p=point_from_records(records,prior,split,weight,budget,scope)
                if set(p['utility'])!=set(UTILITY_TASKS) or set(p['gains'])!=set(FORBIDDEN):raise ValueError(f'incomplete selected endpoints {seed}/{condition}/{scope}/{budget}')
                points[seed,condition,split,weight,budget,scope]=p
    for (seed,c,split,w,b,scope),point in points.items():
        h=points[seed,'H',split,w,b,scope]
        common={'seed':seed,'condition':c,'split':split,'weight':w,'budget':b,'scope':scope}
        kinds={'utility_loss':point['utility'],'attack_loss':point['losses'],'absolute_recovery':point['gains'],
               'additional_recovery':{e:v-h['gains'][e] for e,v in point['gains'].items()},
               'utility_gain_vs_H':{t:h['utility'][t]-v for t,v in point['utility'].items()}}
        for kind,values in kinds.items():
            flat.extend({**common,'kind':kind,'endpoint':e,'value':float(v),'coverage_complete':point['coverage'].get(e)} for e,v in values.items())
    sources[str(historical/'comparison_rules.json')]=sha(historical/'comparison_rules.json')
    return points,raw,flat,candidate,classes,parents,rules,sources


def comparisons(points,parents,rules):
    paired=[];matches=[];vectors=[];source=[]
    for (seed,condition,split,w,b,scope),point in points.items():
        if b==360 and scope==MAIN_SCOPE:
            check=source_check(point,parents[seed,split,w])
            source.extend({'seed':seed,'condition':condition,'split':split,'weight':w,'task':t,**r,'all_source_pass':check['pass']} for t,r in check['tasks'].items())
    for left,right in PAIRS:
        for seed,split,w,b,scope in itertools.product(range(3),('validation','test'),WEIGHTS,(120,360),SCOPES):
            lk=(seed,left,split,w,b,scope);rk=(seed,right,split,w,b,scope)
            if lk not in points or rk not in points:continue
            a,z=points[lk],points[rk];common={'seed':seed,'left':left,'right':right,'split':split,'weight':w,'budget':b,'scope':scope}
            for kind in ('utility','gains'):
                paired.extend({**common,'kind':kind,'endpoint':e,'difference':v-z[kind][e]} for e,v in a[kind].items())
            for panel,tasks in rules['panels'].items():
                for delta in rules['delta_values']:
                    vector=directional(a,z,tasks,delta)
                    vectors.append({**common,'panel':panel,'delta':delta,**vector})
                    for attr in ('SEX','RAC1P'):
                        d=evaluate_pair(a,z,parents[seed,split,w],tasks,delta,attr)
                        matches.append({**common,'panel':panel,'delta':delta,'attribute':attr,
                            **{k:d[k] for k in ('both_source_feasible','utility_close','utility_directional','gain_difference','strict_gain_improvement','qualifies_close','qualifies_directional','assessment_close','assessment_directional','exclusion_reasons')}})
    return paired,matches,vectors,source


def original_headroom(out,original_root,points):
    rows=[];context=[];hashes={}
    for seed in range(3):
        path=original_root/f'results/redesign_20260908_acs_protection_v1/seed_{seed}/metrics.json'
        records=read(path)['raw_metrics'];hashes[str(path)]=sha(path)
        selected={(r['release'],r['target']):r for r in records if r['role']=='transfer' and r['selected']}
        for split,w in itertools.product(('validation','test'),WEIGHTS):
            score=score_key(split,w)
            pca={t:selected['E_pca',t][score]['log_loss'] for t in UTILITY_TASKS}
            banks={c:selected[c,'same_residence'][score]['log_loss'] for c in ('B_rich_bank','C_tree_bank')}
            for condition in CONDITIONS:
                point=points.get((seed,condition,split,w,360,MAIN_SCOPE))
                if point is not None:rows.append({'seed':seed,'condition':condition,'split':split,'weight':w,**utility_criteria(point['utility'],pca,banks)})
            for c in ('E_pca','E_pca_leace','B_rich_bank','C_tree_bank'):
                context.extend({'seed':seed,'condition':c,'split':split,'weight':w,'task':t,'loss':selected[c,t][score]['log_loss'],'audit_scope':'historical120/context only'} for t in UTILITY_TASKS)
    return rows,context,hashes


def save_loss_vectors(path,loss_cache,uniform_cache):
    unique={};keys=[];ids=[]
    for (tag,prediction_key),values in loss_cache.items():
        values=np.ascontiguousarray(values,dtype=np.float64)
        digest=hashlib.sha256(str(values.shape).encode()+values.tobytes()).hexdigest()
        unique.setdefault('loss/'+digest,values)
        keys.append(tag+'/'+prediction_key);ids.append(digest)
    np.savez_compressed(path,**unique,reference_keys=np.asarray(keys),reference_loss_ids=np.asarray(ids),**{'branch/'+pool:u for pool,u in uniform_cache.items()})
    return {'unique_loss_vectors':len(unique),'loss_references':len(keys),'branch_vectors':len(uniform_cache),'sha256':sha(path)}


def stochastic_report(out,original_root,points,raw):
    rows=[];route=[];mixed_points={};checks=[];hashes={};loss_archives=[]
    for seed in range(3):
        if (seed,'H') not in raw:continue
        labels_path=out/f'seed_{seed}'/'evaluation_labels.npz'
        split_path=original_root/f'results/redesign_20260909_acs_fixed_predictions_v1/seed_{seed}/split_rows.npz'
        hashes[str(labels_path)]=sha(labels_path);hashes[str(split_path)]=sha(split_path)
        with np.load(labels_path) as labels,np.load(split_path) as rawrows,np.load(out/f'seed_{seed}'/'H/predictions.npz') as hp:
            for condition in WITHHOLD:
                if (seed,condition) not in raw:continue
                with np.load(out/f'seed_{seed}'/condition/'predictions.npz') as ap:
                    loss_cache={};uniform_cache={}
                    for split,budget,scope in itertools.product(('validation','test'),(120,360),SCOPES):
                        anchor=points[seed,'H',split,'unweighted',budget,scope];augment=points[seed,condition,split,'unweighted',budget,scope]
                        for endpoint,hrow in anchor['selected'].items():
                            arow=augment['selected'][endpoint];role=hrow['role'];target=hrow['target']
                            pool=('downstream_validation' if role=='utility' else 'attacker_validation') if split=='validation' else 'test'
                            if pool not in uniform_cache:uniform_cache[pool]=branch_uniforms(rawrows[pool],seed,condition)
                            valid=np.asarray(labels[pool+'/'+target])>=0
                            if len(rawrows[pool])!=len(valid):raise ValueError('raw-row/label alignment failure')
                            predictions=[]
                            for tag,row,archive in [('H',hrow,hp),(condition,arow,ap)]:
                                pk=f"{role}/{row['view']}/{target}/{row['audit_budget']}/{row['candidate_id']}/{split}"
                                cache_key=(tag,pk)
                                if cache_key not in loss_cache:
                                    losses,mask=person_losses(labels[pool+'/'+target],archive[pk]);loss_cache[cache_key]=losses
                                    for weight in WEIGHTS:
                                        w=labels['weights/'+pool][mask] if weight=='person_weighted' else None
                                        actual=float(np.average(losses,weights=w));saved=row['scores'][score_key(split,weight)]['log_loss']
                                        if abs(actual-saved)>2e-10:raise ValueError(f'selected per-person score mismatch {seed}/{tag}/{pk}/{weight}: {actual-saved}')
                                        checks.append({'seed':seed,'condition':tag,'role':role,'view':row['view'],'target':target,'budget':row['audit_budget'],'candidate_id':row['candidate_id'],'split':split,'weight':weight,'replay_abs_error':abs(actual-saved)})
                                predictions.append(loss_cache[cache_key])
                            h_loss,a_loss=predictions;uniforms=uniform_cache[pool][valid]
                            for p in P_VALUES:
                                expected,sampled=withholding_losses(h_loss,a_loss,p,uniforms)
                                for weight in WEIGHTS:
                                    weights=labels['weights/'+pool][valid] if weight=='person_weighted' else None
                                    avg=lambda x:float(np.average(x,weights=weights))
                                    mean,sample=avg(expected),avg(sampled)
                                    name=f'withhold_{condition}_p{p:g}';key=(seed,name,split,weight,budget,scope)
                                    point=mixed_points.setdefault(key,{'utility':{},'gains':{},'losses':{},'coverage':{},'assessable':anchor['assessable'] and augment['assessable']})
                                    base={'seed':seed,'condition':condition,'mechanism':name,'p':p,'split':split,'weight':weight,'budget':budget,'scope':scope,'role':role,'view':hrow['view'],'target':target,
                                          'H_candidate':hrow['candidate_id'],'augmented_candidate':arow['candidate_id'],'expected_loss':mean,'sampled_loss':sample,'sampled_minus_expected':sample-mean,
                                          'H_loss':avg(h_loss),'augmented_loss':avg(a_loss),'valid_rows':int(valid.sum()),'sampled_augmented_fraction':avg((uniforms<p).astype(float))}
                                    if role=='utility':point['utility'][target]=mean;base['utility_gain_vs_H']=avg(h_loss)-mean
                                    else:
                                        ep=hrow['view']+'/'+target;hpoint=points[seed,'H',split,weight,budget,scope]
                                        prior=hpoint['gains'][ep]+hpoint['losses'][ep]
                                        point['losses'][ep]=mean;point['gains'][ep]=prior-mean;point['coverage'][ep]=anchor['coverage'][ep] and augment['coverage'][ep]
                                        base.update(absolute_recovery=prior-mean,additional_recovery=avg(h_loss)-mean,coverage_complete=point['coverage'][ep])
                                    rows.append(base)
                    loss_folder=out/f'seed_{seed}'/'withholding_losses';loss_folder.mkdir(exist_ok=True)
                    archive_path=loss_folder/(condition+'.npz')
                    archive_record=save_loss_vectors(archive_path,loss_cache,uniform_cache)
                    loss_archives.append({'seed':seed,'condition':condition,'local_only_path':str(archive_path.relative_to(out)),**archive_record,'published':False})
                    for pool,u in uniform_cache.items():
                        route.append({'seed':seed,'condition':condition,'pool':pool,'n_rows':len(u),'raw_rows_sha256':hashlib.sha256(np.asarray(rawrows[pool]).tobytes()).hexdigest(),
                                      'uniform_sha256':hashlib.sha256(u.tobytes()).hexdigest(),'same_person_branch_all_views_targets_budgets_scopes':True})
    # Exact enumeration of the independent two-person Bernoulli fixture and one sampled draw.
    toy_h=np.array([.1,3.]);toy_a=np.array([2.,.2]);p=.25
    enumeration=np.zeros(2)
    for b0,b1 in itertools.product((0,1),repeat=2):
        prob=(p if b0 else 1-p)*(p if b1 else 1-p)
        enumeration+=prob*np.where([b0,b1],toy_a,toy_h)
    expected,sampled=withholding_losses(toy_h,toy_a,p,np.random.default_rng(20264910).random(2))
    np.testing.assert_allclose(expected,enumeration,atol=1e-15)
    diagnostics={'branch_rule':'SHA256(spectral-withholding-v1|seed|control|raw_row), first8bytes big-endian /2**64; augmented iff u<p',
                 'branch_visible_to':['A','AB'],'B_wire_exactly_original_coverage_no_indicator':True,'request_resampling':False,'branch_independent_of_labels':True,'probability_averaging':False,
                 'expected_loss_formula':'weighted average of aligned (1-p)*person_loss_H+p*person_loss_augmented',
                 'toy_enumerated_expected':enumeration.tolist(),'toy_formula_expected':expected.tolist(),'toy_rng_sampled_losses':sampled.tolist(),
                 'max_selected_score_replay_abs_error':max((r['replay_abs_error'] for r in checks),default=0.),'selected_score_replay_checks':len(checks),'route_identity':route,'local_loss_archives':loss_archives}
    return rows,mixed_points,checks,diagnostics,hashes


def withholding_comparisons(points,mixed,parents,rules):
    rows=[]
    for (seed,name,split,weight,budget,scope),right in mixed.items():
        if split!='test' or budget!=360 or scope!=MAIN_SCOPE:continue
        for arm in SPECTRAL:
            left=points.get((seed,arm,split,weight,budget,scope))
            if left is None:continue
            for panel,tasks in rules['panels'].items():
                for delta in rules['delta_values']:
                    rows.append({'seed':seed,'left':arm,'right':name,'split':split,'weight':weight,'budget':budget,'scope':scope,'panel':panel,'delta':delta,
                                 'both_source_feasible':source_check(left,parents[seed,split,weight])['pass'] and source_check(right,parents[seed,split,weight])['pass'],
                                 **directional(left,right,tasks,delta),
                                 'utility_close':all(abs(left['utility'][t]-right['utility'][t])<=delta+1e-12 for t in tasks),
                                 'withholding_dominates':directional(right,left,tasks,delta)['dominates'],
                                 'withholding_vector':directional(right,left,tasks,delta)})
    return rows


def plots(out,points,mixed,status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'tradeoff_figures';folder.mkdir(exist_ok=True);files=[]
    endpoints=('AB/SEX','AB/RAC1P','A/RAC1P','A/SEX')
    for seed,weight,budget,scope in itertools.product(range(3),WEIGHTS,(120,360),SCOPES):
        key=(seed,'H','test',weight,budget,scope)
        if key not in points:continue
        h=points[key];fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
        for ax,endpoint in zip(axes.flat,endpoints):
            ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
            for condition,color in zip(WITHHOLD,('tab:gray','tab:orange','tab:green','tab:purple')):
                values=[mixed.get((seed,f'withhold_{condition}_p{p:g}','test',weight,budget,scope)) for p in P_VALUES]
                if not all(v is not None for v in values):continue
                x=[h['utility']['same_residence']-v['utility']['same_residence'] for v in values]
                y=[v['gains'][endpoint]-h['gains'][endpoint] for v in values]
                ax.plot(x,y,'o--',lw=1,ms=4,color=color,label=f'{condition} withholding p=0,.25,.5,.75,1')
            for family,names,color in [('M',('S0','M025','M1'),'tab:blue'),('L',('L025','L1','L2'),'tab:red'),('C',('C025','C1'),'tab:cyan')]:
                found=[(name,points.get((seed,'spectral_'+name,'test',weight,budget,scope))) for name in names]
                found=[(name,p) for name,p in found if p is not None]
                x=[h['utility']['same_residence']-p['utility']['same_residence'] for _,p in found]
                y=[p['gains'][endpoint]-h['gains'][endpoint] for _,p in found]
                ax.plot(x,y,'s-',color=color,lw=1,ms=5,label=f'spectral {family} fixed sequence')
                for name,xx,yy in zip([name for name,_ in found],x,y):ax.annotate(name,(xx,yy),xytext=((-23,8) if name=='C025' else (-17,5) if name=='C1' else (4,-10) if name in ('M025','L1','L2') else (4,4)),textcoords='offset points',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.55,'pad':.3})
            j=points.get((seed,'J','test',weight,budget,scope))
            if j is not None:ax.scatter([h['utility']['same_residence']-j['utility']['same_residence']],[j['gains'][endpoint]-h['gains'][endpoint]],marker='*',s=100,color='black',label='frozen J')
            ax.set_title(endpoint);ax.set_xlabel('Residence log-loss gain versus H (higher is better)');ax.set_ylabel('Additional recovery versus H (lower is better)');ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle(f"{'INCOMPLETE PREVIEW — ' if not status['complete'] else ''}Seed {seed} · {weight} · budget {budget}\n{scope} · development, descriptive only",fontsize=12)
        stem=f'seed{seed}_{weight}_{budget}_{scope}'
        for suffix in ('png','pdf'):
            file=folder/(stem+'.'+suffix);fig.savefig(file,dpi=140);files.append(str(file.relative_to(out)))
        plt.close(fig)
    return files


def surrogate_rows(training,heldout,seed,arms=SPECTRAL):
    result=[]
    for arm,role in itertools.product(arms,('local','coalition')):
        reference=training['penalties'][role];denominator=reference['denominator']
        pools={'representation_fit':reference['attributes'],**{pool:values[role] for pool,values in heldout.items()}}
        for pool,attributes in pools.items():
            grouped=[]
            for attribute,details in attributes.items():
                trace=reference['attributes'][attribute]['raw_trace']
                raw=sum(c['output_squared_norms'][arm] for c in details['classes'])
                normalized=raw/trace if trace>1e-12 else (0. if pool=='representation_fit' else None)
                row={'seed':seed,'condition':arm,'pool':pool,'role':role,'attribute':attribute,
                     'raw_projected_moment':raw,'training_attribute_trace':trace,
                     'normalized_projected_moment':normalized,'normalization':'fixed training attribute trace; then fixed role denominator',
                     'fixed_role_denominator':denominator,'zero_training_trace':trace<=1e-12,
                     'valid_rows':details['valid_rows'],'unsupported_classes':details['unsupported_classes'],
                     'prediction_source':'household_grouped_OOF' if pool=='representation_fit' else 'equal_ensemble_three_training_fold_models',
                     'claim_scope':'finite residual moments, not conditional privacy'}
                result.append(row);grouped.append(row)
            result.append({'seed':seed,'condition':arm,'pool':pool,'role':role,'attribute':'__mean__',
                'raw_projected_moment':sum(r['raw_projected_moment'] for r in grouped)/denominator,
                'training_attribute_trace':None,
                'normalized_projected_moment':sum(r['normalized_projected_moment'] for r in grouped)/denominator if all(r['normalized_projected_moment'] is not None for r in grouped) else None,
                'normalization':'normalize each attribute by fixed training trace before fixed role mean',
                'fixed_role_denominator':denominator,'zero_training_trace':any(r['zero_training_trace'] for r in grouped),
                'valid_rows':None,'unsupported_classes':{r['attribute']:r['unsupported_classes'] for r in grouped},
                'prediction_source':grouped[0]['prediction_source'],'claim_scope':'finite residual moments, not conditional privacy'})
    return result


def collect_surrogates(out,points):
    rows=[];hashes={}
    for seed in range(3):
        training_path=out/f'seed_{seed}'/'matrix_diagnostics.json'
        heldout_path=out/f'seed_{seed}'/'heldout_moments.json'
        if not training_path.exists() or not heldout_path.exists():continue
        hashes[str(training_path)]=sha(training_path);hashes[str(heldout_path)]=sha(heldout_path)
        seed_rows=surrogate_rows(read(training_path),read(heldout_path),seed)
        for row in seed_rows:
            view='A' if row['role']=='local' else 'AB';endpoint=view+'/'+row['attribute']
            for weight in WEIGHTS:
                point=points.get((seed,row['condition'],'test',weight,360,MAIN_SCOPE));h=points.get((seed,'H','test',weight,360,MAIN_SCOPE))
                row['empirical_test_additional_recovery_'+weight]=point['gains'][endpoint]-h['gains'][endpoint] if point is not None and h is not None and endpoint in point['gains'] else None
        rows.extend(seed_rows)
    return rows,hashes


def _mean_value(means,condition,weight,kind,endpoint,scope=MAIN_SCOPE):
    rows=[r for r in means if r['condition']==condition and r['weight']==weight and r['kind']==kind and r['endpoint']==endpoint and r['scope']==scope and r['split']=='test' and r['budget']==360]
    return rows[0] if rows else None


def _cell(record):
    if not record:return 'missing'
    return f"{record['mean']:.6f} ± {record['sample_sd']:.6f}"+(f" (n={record['n_seeds']})" if record['n_seeds']!=3 else '') if record['sample_sd'] is not None else f"{record['mean']:.6f} (n={record['n_seeds']})"


def negative_increment_counts(flat):
    groups=defaultdict(list)
    for row in flat:
        if row['kind']!='additional_recovery' or row['split']!='test' or row['condition']=='H':continue
        group='spectral' if row['condition'] in SPECTRAL else 'historical_augmented'
        groups[group,row['weight'],row['budget'],row['scope']].append(row)
    return [{'system_group':group,'weight':weight,'budget':budget,'scope':scope,'split':'test',
             'negative_count':sum(r['value']<0 for r in rows),'below_minus_1e12_count':sum(r['value']< -1e-12 for r in rows),
             'total_count':len(rows),'n_seeds':len({r['seed'] for r in rows}),
             'minimum_signed_additional_recovery':min(r['value'] for r in rows),
             'negative_by_endpoint':{e:sum(r['value']<0 for r in rows if r['endpoint']==e) for e in FORBIDDEN},
             'meaning':'validation-selected augmented attack has worse development log loss than validation-selected H attack; no clipping or test reselection'}
            for (group,weight,budget,scope),rows in groups.items()]


def decision_report(out,status,decision,means,source,half,withhold_compare,paired,surrogates,negative_counts):
    lines=['# Residual spectral development decision','',f"Status: **{status['status']}**; {status['completed_units']}/42 evaluation units. This is the already studied California 2018 cohort, not independent confirmation. All 24 maps were frozen before downstream evaluation. Spectral auxiliaries are float64; historical neural float32 auxiliaries are promoted losslessly. Wires are float64 and coordinate counts are matched where r=16, but underlying numerical precision is not matched. Anchors are never requantized. No privacy guarantee follows from these empirical attacks or finite moments.",'']
    if status['complete']:
        benefit=sum(v['residence_at_least_01_all_seed_weights'] for v in decision['checks'].values())
        lines += [f'Residual preservation added residence capability beyond H: {benefit}/8 spectral recipes exceed the .01 gain threshold in every seed and both weightings. It did not establish a robust method advance over frozen J: no recipe passes the full source-and-sensitive vector requirements, no recipe is nominated, and coalition conditioning fails the fixed coordination criterion.',
                  'The exact source-service probability vectors remain preserved on every release. Failures below concern separately fitted source probes versus the legacy PCA32 allowance; they do not mean the immutable source service changed. [PER_SEED.csv.gz](PER_SEED.csv.gz) provides the complete per-seed, endpoint, scope, budget and weighting table.','']
    lines+=['## 1. Did residual preservation add useful capability beyond H?','',
            'The table reports mean ± sample SD over available original seeds; complete results require all three, and partial values show their n. The .01 improvement over H and original PCA/rich-bank half-headroom are distinct criteria; source allowance requires each original PCA source-probe loss + .01, checked separately on validation and development.','',
            '| Arm | Unweighted residence gain vs H | PWGTP residence gain vs H | .01 in every seed/weight |','|---|---:|---:|---|']
    for c in CONDITIONS:
        check=decision.get('checks',{}).get(c,{})
        lines.append(f"| {c} | {_cell(_mean_value(means,c,'unweighted','utility_gain_vs_H','same_residence'))} | {_cell(_mean_value(means,c,'person_weighted','utility_gain_vs_H','same_residence'))} | {check.get('residence_at_least_01_all_seed_weights','reference' if c in HIST else 'incomplete')} |")
    lines+=['','Separate legacy criteria (development headroom count and source preservation on both validation and development):','','| Arm | Weight | Source pass seeds | Original half-headroom pass seeds |','|---|---|---:|---:|']
    for c in CONDITIONS:
        for weight in WEIGHTS:
            present=sorted({r['seed'] for r in source if r['condition']==c and r['weight']==weight})
            source_count=sum(all(r['pass'] is True for r in source if r['condition']==c and r['weight']==weight and r['seed']==seed) for seed in present)
            hs=[r for r in half if r['condition']==c and r['weight']==weight and r['split']=='test']
            headroom_count=sum(r['residential_retention']['pass'] is True for r in hs)
            lines.append(f'| {c} | {weight} | {source_count}/{len(present)} | {headroom_count}/{len(hs)} |')
    lines+=['','Original half-headroom results are in [ORIGINAL_CRITERIA.csv.gz](ORIGINAL_CRITERIA.csv.gz); no H denominator replaces PCA32. Source failures cannot cancel across tasks, seeds or weighting.','',
            '## 2. Did a spectral method improve on J and simpler alternatives?','',
            f"Fixed-protocol nomination: **{decision.get('nominee') or ('none' if status['complete'] else 'no decision on incomplete matrix')}**. Qualifying recipes: {decision.get('qualifying_arms',[])}. No mean-based or per-seed recipe selection is used.",
            'All eight recipes versus J/H/E/A0, historical J versus controls, and fixed main comparisons are in [PAIRED.csv.gz](PAIRED.csv.gz) and [PAIRED_AGGREGATE.csv.gz](PAIRED_AGGREGATE.csv.gz). Close matches preserve all four original panels and deltas; directional vectors retain each A/B/AB SEX/race role separately.','',
            '## 3. Did coalition conditioning beat equal-setting and equal-mass local controls?','',
            f"Fixed C1 versus L1 and C1 versus L2 coordination criterion: **{decision.get('coordination_supported','incomplete')}**. The same all-seed/both-weight vector rule and one common strict endpoint apply to each comparison; C025 versus L025 is a separately reported sensitivity comparison.",
            '[DIRECTIONAL_VECTORS.csv.gz](DIRECTIONAL_VECTORS.csv.gz) exposes local race differences directly. A coalition gain does not compensate for worse local race. [DECISION.json](DECISION.json) records every threshold and failing component.','',
            '## 4. Did it survive nonlinear attacks, both weightings and local race?','',
            'The common kernel expanded catch-up360 scope is used for the decision, including bounded Gaussian RFF ridge attackers and all legal historical ancestors. Original no-kernel scopes and independent-only scopes remain separate. A spectral method has no historical saved observer; unequal inherited history is disclosed. Every candidate, including every fixed ancestor, has its own score row; no development-best ancestor is substituted. Absent-class recall/AUROC and full-schema protection remain unassessable. Finite observed-distribution log-loss comparisons still follow the prospective numerical vector rule; full class coverage is reported separately, never added as an absolute nomination gate.','',
            '| Arm | Weight | A/race additional recovery | AB/race additional recovery | AB/SEX additional recovery |','|---|---|---:|---:|---:|']
    for c in ('J',)+SPECTRAL:
        for weight in WEIGHTS:
            lines.append('| '+ ' | '.join([c,weight]+[_cell(_mean_value(means,c,weight,'additional_recovery',e)) for e in ('A/RAC1P','AB/RAC1P','AB/SEX')])+' |')
    lines+=['','Validation-selected attacks sometimes generalize worse than the H attack. Negative signed additional recovery is retained exactly, without clipping or development reselection. Counts below cover all eleven forbidden roles per seed and condition; H is excluded from its own comparisons. Parenthesized counts are below -1e-12, separating numerical roundoff from strictly negative values.','','| Scope | Budget | Weight | Spectral negative / total (below -1e-12) | Historical augmented negative / total (below -1e-12) |','|---|---:|---|---:|---:|']
    neg_index={(r['system_group'],r['weight'],r['budget'],r['scope']):r for r in negative_counts}
    for scope,budget,weight in itertools.product(SCOPES,(120,360),WEIGHTS):
        cells=[]
        for group in ('spectral','historical_augmented'):
            r=neg_index.get((group,weight,budget,scope))
            cells.append(f"{r['negative_count']}/{r['total_count']} ({r['below_minus_1e12_count']})" if r else 'missing')
        lines.append('| '+' | '.join([scope,str(budget),weight]+cells)+' |')
    lines += ['','[NEGATIVE_INCREMENT_COUNTS.csv.gz](NEGATIVE_INCREMENT_COUNTS.csv.gz) also gives endpoint-specific negative counts. These are descriptive generalization differences of frozen validation selections, not evidence that removing information improves an optimal attacker.']
    lines+=['','Training versus heldout residual moments use the same fixed training attribute-trace normalization and fixed local/coalition denominators 3/2. Training uses grouped OOF nuisances; heldout uses the equal ensemble of three frozen nuisance models, without refitting. Values are finite-surrogate diagnostics, separate from the empirical recovery table; a low average does not establish privacy, and unknown/zero training normalizers stay explicit.','','| Arm | Role | Train normalized moment | Development normalized moment |','|---|---|---:|---:|']
    for c,role in itertools.product(('spectral_S0','spectral_C1','spectral_L1','spectral_L2'),('local','coalition')):
        cells=[]
        for pool in ('representation_fit','test'):
            values=[r['normalized_projected_moment'] for r in surrogates if r['condition']==c and r['role']==role and r['pool']==pool and r['attribute']=='__mean__' and r['normalized_projected_moment'] is not None]
            cells.append(f'{statistics.mean(values):.6f} (n={len(values)})' if values else 'undefined')
        lines.append('| '+' | '.join([c,role]+cells)+' |')
    lines+=['','[SURROGATE_DIAGNOSTICS.csv.gz](SURROGATE_DIAGNOSTICS.csv.gz) reports every arm, class-support limitations, attribute and fixed role mean for representation fitting, source validation and development, with raw projected norms and separately labeled empirical recovery.']
    for comparator,gate in decision.get('coordination',{}).items():
        lines.append(f"C1 versus {comparator}: vector nonworsening={gate.get('vector_nonworsening')}; residence strict in all six seed/weight pairs={gate.get('strict_residence_all_seed_weights')}; fixed sensitive endpoints strict in all six={gate.get('shared_strict_sensitive_roles',[])}. Each comparator must pass independently; the strict endpoint may differ between comparators.")
    lines+=['','Fixed coordination comparisons, mean paired difference (left minus right; lower is better for every displayed component), common kernel expanded catch-up360:','','| Left / right | Weight | Residence loss | AB/SEX recovery | AB/race recovery | A/race recovery |','|---|---|---:|---:|---:|---:|']
    for left,right in MAIN_PAIRS:
        for weight in WEIGHTS:
            cells=[]
            for kind,endpoint in [('utility','same_residence'),('gains','AB/SEX'),('gains','AB/RAC1P'),('gains','A/RAC1P')]:
                values=[r['difference'] for r in paired if r['left']==left and r['right']==right and r['split']=='test' and r['weight']==weight and r['budget']==360 and r['scope']==MAIN_SCOPE and r['kind']==kind and r['endpoint']==endpoint]
                cells.append(f'{statistics.mean(values):.6f} (n={len(values)})' if values else 'missing')
            lines.append('| '+' | '.join([left+' / '+right,weight]+cells)+' |')
    lines+=['','The original historical J residence gains versus H were approximately .021528 unweighted and .019516 PWGTP; its additional AB/SEX recovery was approximately .001109 and .000740, respectively. These values belong to the original audit context. Recomputed original versus common-kernel comparisons follow:','',
            '| Weight | J original expanded catch-up AB/SEX | J common kernel expanded catch-up AB/SEX |','|---|---:|---:|']
    for weight in WEIGHTS:lines.append(f"| {weight} | {_cell(_mean_value(means,'J',weight,'additional_recovery','AB/SEX','expanded_catchup'))} | {_cell(_mean_value(means,'J',weight,'additional_recovery','AB/SEX'))} |")
    lines+=['','[PER_CLASS.csv.gz](PER_CLASS.csv.gz) and [ALL_CANDIDATES.csv.gz](ALL_CANDIDATES.csv.gz) retain fixed class support, recall/AUC availability and complete candidate scores. The original and kernel scopes are not pooled.','',
            '## 5. Did actual withholding explain the trade-off?','',
            'H remains visible. A label-independent Bernoulli branch is visible to A and AB and fixed per person/release with the same draw for A/AB. B remains exactly the original coverage vector with no branch indicator; routed B predictors are identical to H. Each p=0,.25,.5,.75,1 has routed saved H/augmented predictors; expected per-person log losses are mixed before averaging, never probabilities or channels. Utility is an average over people with unequal branches. No p is selected.',
            '[WITHHOLDING.csv.gz](WITHHOLDING.csv.gz) gives all five utility and eleven forbidden roles, both budgets, six scopes, both weightings and both splits; sampled routing and score replay appear in [WITHHOLDING_VALIDATION.json](WITHHOLDING_VALIDATION.json). Aligned person-loss vectors and branch uniforms are retained in local-only compressed archives whose hashes are listed there; individual rows are not published. [WITHHOLDING_COMPARISONS.csv.gz](WITHHOLDING_COMPARISONS.csv.gz) gives componentwise spectral comparisons at every fixed p.','']
    if withhold_compare:
        main=[r for r in withhold_compare if r['panel']=='full_authorized' and r['delta']==.001]
        for c in SPECTRAL:
            rs=[r for r in main if r['left']==c]
            wins=sum(r['dominates'] and r['both_source_feasible'] for r in rs)
            lines.append(f'- Spectral → withholding, {c}: {wins}/{len(rs)} seed/weight/fixed-p comparisons satisfy the full-vector directional criterion at delta=.001; this count is descriptive and is not a selected p or an all-seed dominance claim.')
            all_fixed=[]
            for mechanism in sorted({r['right'] for r in rs}):
                fixed=[r for r in rs if r['right']==mechanism]
                if len(fixed)==6 and all(r['withholding_dominates'] and r['both_source_feasible'] for r in fixed):all_fixed.append(mechanism)
            lines.append(f'  Fixed withholding mechanisms that directionally dominate {c} in all three seeds and both weightings: {all_fixed if status["complete"] else "incomplete, no all-seed conclusion"}. Every listed p remains descriptive; none is nominated.')
    if status['complete'] and withhold_compare:
        main=[r for r in withhold_compare if r['panel']=='full_authorized' and r['delta']==.001]
        forward=sum(r['dominates'] and r['both_source_feasible'] for r in main)
        reverse=sum(r['withholding_dominates'] and r['both_source_feasible'] for r in main)
        lines += [f'Plain result under the full source-feasible five-utility/six-sensitive-role directional rule at delta=.001: spectral dominates withholding in {forward}/{len(main)} individual fixed comparisons; withholding dominates spectral in {reverse}/{len(main)}. No fixed withholding mechanism dominates a spectral recipe across all seeds and both weightings. The curves therefore do not establish a robust full-vector advantage in either direction; higher residence utility and lower sensitive recovery remain separate components.']
    lines+=['','Every per-seed trade-off figure has four sensitive endpoints, all spectral arms and all withholding curves; PNG and PDF variants are listed in [FIGURES.json](FIGURES.json).','',
            '## 6. What limitation remains, and what one next study resolves it?','']
    if not status['complete']:
        lines+=['The matrix is incomplete. No recipe or next-study decision is made until all 42 frozen evaluations are complete.']
    elif decision.get('nominee'):
        lines += [f"Only **{decision['nominee']}** is nominated for later transport as development selection. The remaining limitation is generalization beyond this repeatedly studied cohort and finite attacker family. One next study is the predeclared admitted-year transport evaluation with this exact frozen recipe, frozen and fresh attacks on separate household fitting/selection/final partitions, and no recipe sweep or final-outcome access during admission."]
    else:
        failed_source=[c for c,d in decision['checks'].items() if not d['source_all_validation_and_development']]
        failed_res=[c for c,d in decision['checks'].items() if not d['residence_at_least_01_all_seed_weights']]
        failed_j=[c for c,d in decision['checks'].items() if not d['versus_J']['pass']]
        lines += [f"No recipe meets the fixed nomination rule. Source failures: {failed_source}; insufficient all-seed residence gains: {failed_res}; failed J vector/strict comparison: {failed_j}. These are separate limitations, not averaged away.",
                  'One next study is an explicitly labeled limitation study on the admitted California 2017 cohort: preregister the already frozen C1, L2, J and H systems, frozen plus fresh attacks, and fixed H-defined strata; use disjoint household fitting, selection and final partitions to test whether residual-moment blind spots and residence gains transport. This is not a new recipe nomination or a successful-method confirmation, does not expand the grid, and is not launched here. The present 2018 scores remain reused development evidence; reshuffling them would not create independent confirmation.']
    if status['complete'] and not decision.get('nominee'):
        lines+=['','Concrete diagnostic evidence for that limitation:']
        for c in ('spectral_C1','spectral_L2'):
            values={}
            for pool in ('representation_fit','test'):
                terms=[r['normalized_projected_moment'] for r in surrogates if r['condition']==c and r['role']=='local' and r['pool']==pool and r['attribute']=='__mean__' and r['normalized_projected_moment'] is not None]
                values[pool]=statistics.mean(terms) if terms else None
            gains=[_cell(_mean_value(means,c,w,'additional_recovery','A/RAC1P')) for w in WEIGHTS]
            lines.append(f'- {c}: normalized local moment mean train={values["representation_fit"]}, development={values["test"]}; additional local-race recovery unweighted={gains[0]}, PWGTP={gains[1]}. The finite-surrogate and attack columns measure different quantities and are not interchangeable guarantees.')
    lines+=['','Finite trace optimality, exact-source LP feasibility, and empirical attack resistance are different claims. The mathematical counterexamples show why zero first moments are insufficient for unrestricted conditional privacy. [MATHEMATICS.md](MATHEMATICS.md) and [NUMERICAL_VALIDATION.json](NUMERICAL_VALIDATION.json) contain the derivation and synthetic checks. Means/sample SD are descriptive; three seeds are not independent population replications. Independent pre-report review aligned the decision implementation with the prospective finite-log-loss rule: full-schema class support remains a separate limitation, not an additional nomination criterion. No individual rows are published in these aggregate tables.','']
    text='\n'.join(lines)
    for name in ('PER_SEED','MEANS','ALL_CANDIDATES','PER_CLASS','PAIRED','PAIRED_AGGREGATE','UTILITY_MATCHES','DIRECTIONAL_VECTORS','SOURCE_FEASIBILITY','ORIGINAL_CRITERIA','ORIGINAL_CONTEXT','WITHHOLDING','WITHHOLDING_SCORE_REPLAY','WITHHOLDING_COMPARISONS','SURROGATE_DIAGNOSTICS','NEGATIVE_INCREMENT_COUNTS'):
        text=text.replace(']('+name+'.csv.gz)','](PUBLICATION_EVIDENCE/'+name+'.csv.gz)')
    text=text.replace('[NUMERICAL_VALIDATION.json](NUMERICAL_VALIDATION.json)','[NUMERICAL_SUMMARY.json](NUMERICAL_SUMMARY.json)')
    (out/'DECISION_REPORT.md').write_text(text)


def report(out,original_root=ORIGINAL_ROOT,allow_incomplete=False,make_plots=True):
    tick=time.perf_counter();out=Path(out);original_root=Path(original_root)
    status=completion_status(out,allow_incomplete)
    if not status['available']:raise ValueError('no completed scores available for a post-freeze preview')
    points,raw,flat,candidate,classes,parents,rules,sources=load_evidence(out,status,original_root)
    means=summarize(flat,('condition','split','weight','budget','scope','kind','endpoint'))
    paired,matches,vectors,source=comparisons(points,parents,rules)
    pair_means=summarize(paired,('left','right','split','weight','budget','scope','kind','endpoint'),'difference')
    half,context,context_hashes=original_headroom(out,original_root,points);sources.update(context_hashes)
    withholding,mixed,replay,routing,withhold_hashes=stochastic_report(out,original_root,points,raw);sources.update(withhold_hashes)
    withhold_comparisons=withholding_comparisons(points,mixed,parents,rules)
    decision=nominate(points,parents,status['complete'])
    surrogates,surrogate_hashes=collect_surrogates(out,points);sources.update(surrogate_hashes)
    negative_counts=negative_increment_counts(flat)
    for name,rows in [('PER_SEED',flat),('MEANS',means),('ALL_CANDIDATES',candidate),('PER_CLASS',classes),('PAIRED',paired),('PAIRED_AGGREGATE',pair_means),('UTILITY_MATCHES',matches),('DIRECTIONAL_VECTORS',vectors),('SOURCE_FEASIBILITY',source),('ORIGINAL_CRITERIA',half),('ORIGINAL_CONTEXT',context),('WITHHOLDING',withholding),('WITHHOLDING_SCORE_REPLAY',replay),('WITHHOLDING_COMPARISONS',withhold_comparisons),('SURROGATE_DIAGNOSTICS',surrogates),('NEGATIVE_INCREMENT_COUNTS',negative_counts)]:csvout(out/(name+'.csv.gz'),rows)
    write_json(out/'DECISION.json',decision);write_json(out/'WITHHOLDING_VALIDATION.json',routing);write_json(out/'REPORT_INPUT_HASHES.json',sources)
    files=plots(out,points,mixed,status) if make_plots else []
    write_json(out/'FIGURES.json',{'files':files,'scope':'all seeds/weightings/120+360/six scopes, development four-endpoint curves','complete_matrix':status['complete']})
    decision_report(out,status,decision,means,source,half,withhold_comparisons,paired,surrogates,negative_counts)
    write_json(out/'REPORT_COUNTS.json',{**status,'points':len(points),'all_candidate_score_rows':len(candidate),'per_class_rows':len(classes),'selected_endpoint_rows':len(flat),'paired_rows':len(paired),'matching_rows':len(matches),'directional_rows':len(vectors),'withholding_rows':len(withholding),'withholding_comparison_rows':len(withhold_comparisons),'surrogate_diagnostic_rows':len(surrogates),'figure_files':len(files),'elapsed_seconds':time.perf_counter()-tick,'source_code_sha256':sha(__file__),'no_fits':True,'no_raw_person_rows_exported':True})
    return decision


def read_table(path):
    def value(text):
        if text=='':return None
        if text in ('True','False'):return text=='True'
        try:return json.loads(text)
        except (ValueError,TypeError):return text
    with gzip.open(path,'rt',newline='') as f:
        return [{key:value(v) for key,v in row.items()} for row in csv.DictReader(f)]


def refresh_document(out):
    tick=time.perf_counter();out=Path(out);status=completion_status(out)
    tables={name:read_table(out/(name+'.csv.gz')) for name in ('PER_SEED','MEANS','SOURCE_FEASIBILITY','ORIGINAL_CRITERIA','WITHHOLDING_COMPARISONS','PAIRED','SURROGATE_DIAGNOSTICS')}
    negative_counts=negative_increment_counts(tables['PER_SEED'])
    csvout(out/'NEGATIVE_INCREMENT_COUNTS.csv.gz',negative_counts)
    decision=read(out/'DECISION.json')
    decision_report(out,status,decision,tables['MEANS'],tables['SOURCE_FEASIBILITY'],tables['ORIGINAL_CRITERIA'],tables['WITHHOLDING_COMPARISONS'],tables['PAIRED'],tables['SURROGATE_DIAGNOSTICS'],negative_counts)
    counts=read(out/'REPORT_COUNTS.json');counts.update(source_code_sha256=sha(__file__),documentation_refresh_elapsed_seconds=time.perf_counter()-tick,negative_increment_count_rows=len(negative_counts))
    write_json(out/'REPORT_COUNTS.json',counts)
    return {'document_refreshed':True,'figures_preserved':counts['figure_files'],'negative_count_rows':len(negative_counts),'nominee':decision['nominee']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260910_acs_residual_spectral_v1');parser.add_argument('--original-root',type=Path,default=ORIGINAL_ROOT);parser.add_argument('--allow-incomplete',action='store_true');parser.add_argument('--no-plots',action='store_true');parser.add_argument('--doc-only',action='store_true');args=parser.parse_args()
    print(json.dumps(refresh_document(args.out) if args.doc_only else report(args.out,args.original_root,args.allow_incomplete,not args.no_plots),indent=2))
