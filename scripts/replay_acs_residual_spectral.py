"""Independent frozen-object replay; never fit or select from development scores.

Requires all42 systems unless --allow-incomplete is explicit. New reports only.
No experiment fitting/scoring helpers are called. Kernel and spectral inference
are reconstructed literally; ordinary models use the independent saved-state
inference helper established before this study.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from scripts import verify_acs_bottleneck_scores as independent

ROOT=Path(__file__).resolve().parents[1]
NAME='redesign_20260910_acs_residual_spectral_v1'
HIST=('H','E','A0','L025','L20','J')
SPECTRAL=tuple('spectral_'+v for v in ('S0','M025','M1','L025','L1','C025','C1','L2'))
TARGETS=('SEX','RAC1P','income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
TASKS=('same_residence','commute_over20','income_binary','civilian_at_work','public_coverage')
ROLES={'A':('public_coverage','commute_over20','SEX','RAC1P'),
       'B':('income_binary','civilian_at_work','same_residence','SEX','RAC1P'),'AB':('SEX','RAC1P')}
SCOPES=('standard_independent','expanded_independent','expanded_catchup')
SCOPES=SCOPES+tuple('kernel_'+s for s in SCOPES)
ALPHAS=('0.0001','0.01','1.0')
FRESH=('logistic','mlp_0','mlp_1','hist_gb_20','hist_gb_5')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return independent.array_hash(x)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def eligible(candidates,scope):
    """Independent membership from explicit input space and saved provenance."""
    assert scope in SCOPES
    ids=[]
    for cid,m in candidates.items():
        if m.get('diagnostic_only') or 'saved_adversary' in cid:continue
        if not scope.startswith('kernel_') and 'kernel__' in cid:continue
        if not scope.endswith('catchup') and 'catchup' in cid:continue
        if scope.endswith('standard_independent') and m['space']!='wire':continue
        ids.append(cid)
    return sorted(ids)

def verify_selection(candidates,selected,pools):
    assert set(selected)==set(pools)==set(SCOPES)
    for scope in SCOPES:
        ids=eligible(candidates,scope)
        assert ids and ids==sorted(pools[scope]),('candidate pool mismatch',scope)
        winner=min(ids,key=lambda cid:(candidates[cid]['validation_scores']['log_loss'],cid))
        assert selected[scope]==winner,('selection mismatch',scope,winner,selected[scope])

def expected_h_columns(view,width,columns):
    route=[0,1,2,3] if view=='A' else [0,1,2,3,width,width+1] if view=='AB' else [0,1]
    return [route[j] for j in columns] if columns is not None else route

def verify_routes(roles,h_roles,widths,condition):
    """Every sensitive singleton and every available H candidate must survive."""
    n=0
    for view,targets in ROLES.items():
        for target in targets:
            role=view+'/'+target;current=roles[role]
            for space in widths:
                if view=='B' and space=='derived':continue
                for alpha in ALPHAS:
                    assert space+'__kernel__'+alpha in current,(condition,role,space,alpha)
            if condition in SPECTRAL and view!='B':
                assert all('wire__'+cid in current for cid in FRESH)
                assert 'catchup' not in current and 'saved_adversary' not in current
            if condition!='H':
                for cid,source in h_roles[role].items():
                    ident=cid if view=='B' else 'anchor__'+cid
                    actual=current[ident]
                    assert actual['base_candidate_directory']==source['base_candidate_directory']
                    if view!='B':
                        assert actual['space']=='wire'
                        assert actual['projection_columns']==expected_h_columns(view,widths['wire'][0],source['projection_columns'])
                    else:assert actual==source
                    n+=1
    for target in ('SEX','RAC1P'):
        coalition=roles['AB/'+target]
        for view in ('A','B'):
            for cid,source in roles[view+'/'+target].items():
                actual=coalition['inherited_'+view+'__'+cid]
                wa,wb=widths[source['space']]
                route=list(range(wa)) if view=='A' else list(range(wa,wa+wb))
                expected=[route[j] for j in source['projection_columns']] if source['projection_columns'] is not None else route
                assert actual['projection_columns']==expected
                assert actual['space']==source['space'] and actual['base_candidate_directory']==source['base_candidate_directory']
                n+=1
    return n

def literal_basis(basis,x):
    raw=np.column_stack([np.prod(x[:,term],axis=1) for term in basis.terms])
    return ((raw-basis.means)/basis.scales)[:,basis.retained]

def literal_spectral(model,t,a):
    x=(np.asarray(t,np.float64)-model.t_mean)/model.t_scale
    phi=np.column_stack((x,np.sqrt(2/96)*np.cos(x@model.omega+model.phase)))
    q=literal_basis(model.qA,a[:,[1,3]])
    return (phi-q@model.residual_coefficients-model.residual_mean)@model.whitening

def literal_kernel(model,x):
    standardized=(np.asarray(x,np.float64)-model.preprocessing.mean)/model.preprocessing.scale
    f=np.sqrt(2/len(model.phase))*np.cos(standardized@model.omega+model.phase)
    raw=model.prior+(f-model.center)@model.coef
    clipped=np.minimum(1.,np.maximum(1e-12,raw))
    return clipped/clipped.sum(1,keepdims=True)

def literal_labels(frame):
    result={}
    for target in TARGETS:
        y,mask=independent.labels(frame,target)
        y=y.astype(np.int64);y[~mask]=-1;result[target]=y
    return result

class Replay:
    def __init__(self,out,historical_root,allow_incomplete=False):
        self.out,self.root,self.partial=Path(out),Path(historical_root),allow_incomplete
        self.parent=self.root/'results/redesign_20260909_acs_fixed_predictions_v1'
        self.files={};self.models={};self.prediction_cache={};self.h_roles={};self.score_cache={}
        self.r={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope':'independent development-fixture replay; no confirmatory claim',
                'scientific_fits':0,'optimizer_updates':0,'candidate_reselections':0,
                'systems':0,'spectral_systems':0,'historical_systems':0,'prediction_arrays':0,
                'score_dictionaries':0,'scope_selections':0,'legal_routes':0,'anchor_pool_checks':0,
                'spectral_pool_replays':0,'label_arrays':0,'weight_arrays':0,'candidate_fit_hash_checks':0,
                'historical_cached_prediction_checks':0,'B_selected_prediction_checks':0,'systems_detail':[]}
    def checked(self,p):
        p=Path(p);digest=sha(p);key=str(p)
        assert self.files.get(key,digest)==digest,('file changed',key)
        self.files[key]=digest;return p
    def read(self,p):return read(self.checked(p))
    def arrays(self,p):return arrays(self.checked(p))
    def predict(self,path,metadata,x,pool):
        path=Path(path);key=(str(path),pool,ah(x))
        if key not in self.prediction_cache:
            if str(path) not in self.models:
                self.checked(path/'metadata.json')
                if metadata['family']=='kernel_ridge':
                    obj=joblib.load(self.checked(path/'model.joblib'))
                    predict=lambda xx,m=obj:literal_kernel(m,xx)
                else:
                    self.checked(path/'preprocessing.npz')
                    self.checked(path/('model.pt' if metadata['family']=='mlp' else 'model.joblib'))
                    predict,*_=independent.load_inference(path,metadata)
                self.models[str(path)]=predict
            self.prediction_cache[key]=self.models[str(path)](np.ascontiguousarray(x))
        return self.prediction_cache[key]
    def run(self):
        tick=time.perf_counter();gate=self.read(self.out/'RELEASE_MANIFEST.json')
        assert gate['all24_frozen'] and gate['maps']==24 and gate['training_closed']
        assert sha(self.checked(self.out/'PROTOCOL.md'))==gate['protocol_sha256']
        core='experiments/acs_residual_spectral.py'
        assert sha(ROOT/core)==gate['source_hashes'][core]
        available=[(s,c) for s in range(3) for c in HIST+SPECTRAL if (self.out/f'seed_{s}'/c/'complete.json').exists()]
        if not self.partial:assert len(available)==42,('need all42 completed systems',len(available))
        raw=self.root/'data/folktables/2018/1-Year/psam_p06.csv'
        assert sha(self.checked(raw))=='dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0'
        frame=pd.read_csv(raw,usecols=['SEX','RAC1P','PINCP','ESR','PUBCOV','MIG','JWMNP','PWGTP'])
        for s in range(3):
            for name,digest in gate['seeds'][str(s)]['files'].items():assert sha(self.checked(self.out/f'seed_{s}'/name))==digest
            self.models.clear();self.prediction_cache.clear();self.score_cache.clear();self.h_roles={};b_selected={}
            pdest=self.parent/f'seed_{s}';dest=self.out/f'seed_{s}'
            rows=self.arrays(pdest/'split_rows.npz');pca=self.arrays(pdest/'pca.npz');anchors=self.arrays(pdest/'anchors.npz')
            model=joblib.load(self.checked(dest/'maps.joblib'))
            assert set(model.maps)==set(SPECTRAL)
            # Replay all maps regardless of evaluation availability.
            for c in SPECTRAL:
                released=self.arrays(dest/'releases'/c/'releases.npz')
                for pool,t in pca.items():
                    actual=literal_spectral(model,t,anchors[pool+'/A'])@model.maps[c]
                    assert np.array_equal(actual,released['wire/A/'+pool][:,4:]),(s,c,pool,'map inference')
                    self.r['spectral_pool_replays']+=1
            if not any(seed==s for seed,c in available):continue
            labels={pool:literal_labels(frame.iloc[ix]) for pool,ix in rows.items()}
            weights={pool:frame.iloc[ix].PWGTP.to_numpy(float) for pool,ix in rows.items()}
            saved_labels=self.arrays(dest/'evaluation_labels.npz')
            expected_keys={pool+'/'+t for pool in rows if pool!='representation_fit' for t in TARGETS}|{'weights/'+pool for pool in rows if pool!='representation_fit'}
            assert set(saved_labels)==expected_keys
            for pool in rows:
                if pool=='representation_fit':continue
                assert np.array_equal(saved_labels['weights/'+pool],weights[pool]);self.r['weight_arrays']+=1
                for t in TARGETS:
                    assert np.array_equal(saved_labels[pool+'/'+t],labels[pool][t]),(s,pool,t)
                    self.r['label_arrays']+=1
            indices={}
            for role,pool,targets,cap,base in [('utility','downstream_fit',TASKS,2048,1230000),('audit','attacker_fit',TARGETS,4096,1240000)]:
                indices[role]={t:np.random.default_rng(base+100*s+j).permutation(np.flatnonzero(labels[pool][t]>=0))[:cap] for j,t in enumerate(targets)}
            old_indices=self.read(pdest/'indices.json')
            assert old_indices=={'utility':{t:ah(ix) for t,ix in indices['utility'].items()},'attacker':{t:ah(ix) for t,ix in indices['audit'].items()}}
            for c in HIST+SPECTRAL:
                if (s,c) not in available:continue
                print('REPLAY',s,c,flush=True)
                d=dest/c;complete=self.read(d/'complete.json')
                for file,key in [('predictions.npz','prediction_sha256'),('metrics.json','metric_sha256')]:assert sha(self.checked(d/file))==complete[key]
                rp=pdest/'training'/c/'releases.npz' if c in HIST else dest/'releases'/c/'releases.npz'
                wire=self.arrays(rp);widths={space:(wire[space+'/A/attacker_fit'].shape[1],wire[space+'/B/attacker_fit'].shape[1]) for space in ('wire','derived') if space+'/A/attacker_fit' in wire}
                for pool in rows:
                    a,b=wire['wire/A/'+pool],wire['wire/B/'+pool]
                    assert a.dtype==b.dtype==np.float64
                    assert np.array_equal(a[:,:4],anchors[pool+'/A']) and np.array_equal(b,anchors[pool+'/B'])
                    assert np.array_equal(wire['wire/AB/'+pool],np.column_stack((a,b)))
                    self.r['anchor_pool_checks']+=1
                selection=self.read(d/'selection_before_test.json');audit=self.read(d/'audits/audit_selection.json')
                assert selection['audits']==audit['selections']
                for budget,roles in audit['candidates'].items():
                    if c=='H':self.h_roles[budget]=roles
                    assert set(roles)=={v+'/'+t for v,ts in ROLES.items() for t in ts}
                    self.r['legal_routes']+=verify_routes(roles,self.h_roles[budget],widths,c)
                    for role,cs in roles.items():
                        verify_selection(cs,audit['selections'][budget][role],audit['selection_pools'][budget][role]);self.r['scope_selections']+=6
                for role,choice in selection['utility'].items():
                    candidates=selection['utility_metadata'][role]['candidates']
                    assert set(candidates)=={'logistic','mlp'}
                    assert choice==min(candidates,key=lambda cid:(candidates[cid]['validation_scores']['log_loss'],cid))
                records=self.read(d/'metrics.json')['raw_metrics'];predictions=self.arrays(d/'predictions.npz');seen=set()
                expected_records={('audit',*role.split('/'),int(b),cid) for b,roles in audit['candidates'].items() for role,cs in roles.items() for cid in cs}
                expected_records|={('utility',*role.split('/'),None,cid) for role in selection['utility'] for cid in ('logistic','mlp')}
                actual_records=[tuple(row[k] for k in ('role','view','target','audit_budget','candidate_id')) for row in records]
                assert len(actual_records)==len(set(actual_records)) and set(actual_records)==expected_records
                if c in HIST:
                    historical_audit=self.read(pdest/c/'audits/audit_selection.json')
                    for budget,roles in historical_audit['candidates'].items():
                        for role,cs in roles.items():
                            if role.startswith('B/'):continue
                            for cid,old in cs.items():
                                current=audit['candidates'][budget][role][cid]
                                for name in ('base_candidate_directory','space','projection_columns'):
                                    assert current[name]==old[name],(s,c,role,cid,'historical route changed')
                old_predictions=np.load(self.checked(pdest/c/'predictions.npz')) if c in HIST else None
                for row in records:
                    role,v,t,b,cid=(row[k] for k in ('role','view','target','audit_budget','candidate_id'))
                    if role=='audit':
                        meta=audit['candidates'][str(b)][v+'/'+t][cid]
                        path=Path(meta['base_candidate_directory']);base=self.read(path/'metadata.json')
                        assert sha(path/'metadata.json')==meta['base_metadata_sha256']
                        space,cols=meta['space'],meta['projection_columns']
                        wanted=[scope for scope,choice in audit['selections'][str(b)][v+'/'+t].items() if choice==cid]
                    else:
                        meta=selection['utility_metadata'][v+'/'+t]['candidates'][cid]
                        path=(pdest/('H' if v=='B' else c) if c in HIST or v=='B' else d)/'fitted/utility'/v/t/cid
                        base=self.read(path/'metadata.json');assert base==meta
                        space,cols='wire',None
                        wanted=['utility'] if selection['utility'][v+'/'+t]==cid else []
                    assert sorted(row['selected_scopes'])==sorted(wanted)
                    fp='downstream_fit' if role=='utility' else 'attacker_fit';ix=indices[role][t]
                    xf=wire[space+'/'+v+'/'+fp][ix]
                    if cols is not None:xf=xf[:,cols]
                    if base.get('fit_hashes'):
                        assert base['fit_hashes']=={'x':ah(xf),'y':ah(labels[fp][t][ix])},(s,c,role,v,t,cid,'fit identity')
                        self.r['candidate_fit_hash_checks']+=1
                    for split in ('validation','test'):
                        pool=('downstream_validation' if role=='utility' else 'attacker_validation') if split=='validation' else 'test'
                        y=labels[pool][t];valid=y>=0;x=wire[space+'/'+v+'/'+pool][valid]
                        if cols is not None:x=x[:,cols]
                        if split=='validation':assert base['validation_hashes']=={'x':ah(x),'y':ah(y[valid])}
                        actual=self.predict(path,base,x,pool)
                        key=f'{role}/{v}/{t}/{b}/{cid}/{split}';seen.add(key)
                        assert np.array_equal(actual,predictions[key]),(s,c,key,'prediction mismatch')
                        self.r['prediction_arrays']+=1
                        if old_predictions is not None and key in old_predictions:
                            assert np.array_equal(actual,old_predictions[key]);self.r['historical_cached_prediction_checks']+=1
                        for suffix,weight in [('',None),('_person_weighted',weights[pool][valid])]:
                            score_key=(pool,t,ah(actual),weight is not None)
                            if score_key not in self.score_cache:
                                self.score_cache[score_key]=independent.independent_scores(y[valid],actual,9 if t=='RAC1P' else 2,weight)
                            independent.compare(self.score_cache[score_key],row['scores'][split+suffix],f'{s}/{c}/{key}{suffix}')
                            self.r['score_dictionaries']+=1
                        if v=='B':
                            for scope in wanted:
                                bk=(role,t,b,scope,split)
                                if c=='H':b_selected[bk]=actual
                                else:assert np.array_equal(actual,b_selected[bk]);self.r['B_selected_prediction_checks']+=1
                if old_predictions is not None:old_predictions.close()
                assert seen==set(predictions)
                assert not independent.errors,independent.errors[:3]
                # No duplicated direct B fit objects outside H; historical B is external.
                if c!='H':assert not list((d/'audits/fitted').glob('*/B/**/model.*'))
                self.r['systems']+=1;self.r['spectral_systems' if c in SPECTRAL else 'historical_systems']+=1
                self.r['systems_detail'].append({'seed':s,'condition':c,'passed':True,'prediction_arrays':len(predictions)})
        for path,digest in self.files.items():assert sha(path)==digest,('changed during replay',path)
        self.r.update(passed=True,complete=self.r['systems']==42,allow_incomplete=self.partial,
                      files_unchanged=len(self.files),file_manifest_sha256=hashlib.sha256(json.dumps(self.files,sort_keys=True).encode()).hexdigest(),
                      numeric_comparisons=independent.numeric_comparisons,max_score_absolute_error=independent.max_error,
                      runtime_seconds=time.perf_counter()-tick,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      reporting_artifacts_verified=False,reporting_note='Core inference/selection replay. Reporting verification recorded separately when tables are available.')
        return self.r

def csv_rows(path):
    import csv,gzip
    with gzip.open(path,'rt',newline='') as handle:
        yield from csv.DictReader(handle)

def csv_value(value):
    if value=='':return None
    if value in ('True','False'):return value=='True'
    try:return json.loads(value)
    except (ValueError,TypeError):return value

def assert_number(actual,expected,context,tol=2e-10):
    actual=csv_value(actual) if isinstance(actual,str) else actual
    if expected is None:assert actual is None,context
    else:assert actual is not None and abs(float(actual)-float(expected))<=tol,(context,actual,expected)

def verify_reporting(out,root,allow_incomplete=False):
    """Independently recompute aggregate rows, routes, pair rules and nomination."""
    import itertools,statistics
    out,root=Path(out),Path(root);start=time.perf_counter();counts=read(out/'REPORT_COUNTS.json')
    assert counts['complete'] or allow_incomplete
    if not allow_incomplete:
        assert counts['completed_units']==42
        assert {tuple(x) for x in counts['available']}=={(s,c) for s in range(3) for c in HIST+SPECTRAL}
    assert len(counts['available'])==len({tuple(x) for x in counts['available']})
    report={'scope':'independent aggregate/stochastic/comparison verification','scientific_fits':0,
            'candidate_score_rows':0,'selected_endpoint_rows':0,'paired_rows':0,'directional_rows':0,
            'matching_rows':0,'withholding_rows':0,'withholding_comparison_rows':0,'mean_rows':0}
    checked={str(out/'REPORT_COUNTS.json'):sha(out/'REPORT_COUNTS.json')}
    def table(name):
        path=out/(name+'.csv.gz');checked[str(path)]=sha(path);return csv_rows(path)
    scores={};points={};priors={};parent={};allraw={};audit_meta={}
    ancestor=root/'results/redesign_20260909_acs_fixed_predictions_v1'
    rules=read(ancestor/'comparison_rules.json')
    sensitive=tuple(v+'/'+t for v in ('A','B','AB') for t in ('SEX','RAC1P'))
    source=('income_binary','civilian_at_work','public_coverage')
    weights=('unweighted','person_weighted')
    def sk(split,weight):return split+('_person_weighted' if weight=='person_weighted' else '')
    for s in range(3):
        control=root/f'results/redesign_20260908_acs_coalition_v1/seed_{s}/controls/metrics.json'
        checked[str(control)]=sha(control)
        priors[s]={r['target']:r['scores'] for r in read(control)['raw_metrics'] if r['condition']=='prior'}
        for split,weight in itertools.product(('validation','test'),weights):
            parent[s,split,weight]={t:r['scores'][sk(split,weight)] for t,r in rules['original_parent_metric_identity'][str(s)]['tasks'].items()}
    for s,c in counts['available']:
        path=out/f'seed_{s}'/c/'metrics.json';checked[str(path)]=sha(path)
        rows=read(path)['raw_metrics'];allraw[s,c]=rows
        amap=read(out/f'seed_{s}'/c/'audits/audit_selection.json')['candidates']
        for budget,roles in amap.items():
            for role,cs in roles.items():
                for cid,m in cs.items():audit_meta[s,c,int(budget),role,cid]=m
        for row in rows:
            for split,weight in itertools.product(('validation','test'),weights):
                identity=(s,c,row['role'],row['view'],row['target'],row['audit_budget'],row['candidate_id'],split,weight)
                scores[identity]=(row,row['scores'][sk(split,weight)])
        for split,weight,budget,scope in itertools.product(('validation','test'),weights,(120,360),SCOPES):
            pt={'utility':{},'gains':{},'attack':{},'selected':{},'coverage':{}}
            for row in rows:
                score=row['scores'][sk(split,weight)]
                if row['role']=='utility' and 'utility' in row['selected_scopes']:
                    pt['utility'][row['target']]=score['log_loss'];pt['selected']['utility/'+row['target']]=row
                elif row['role']=='audit' and row['audit_budget']==budget and scope in row['selected_scopes']:
                    ep=row['view']+'/'+row['target'];pt['attack'][ep]=score['log_loss']
                    pt['gains'][ep]=priors[s][row['target']][sk(split,weight)]['log_loss']-score['log_loss']
                    pt['selected']['audit/'+ep]=row;pt['coverage'][ep]=score['coverage_complete']
            assert set(pt['utility'])==set(TASKS) and len(pt['gains'])==11
            points[s,c,split,weight,budget,scope]=pt
    candidate_seen=set()
    for row in table('ALL_CANDIDATES'):
        identity=(int(row['seed']),row['condition'],row['role'],row['view'],row['target'],int(row['audit_budget']) if row['audit_budget'] else None,row['candidate_id'],row['split'],row['weight'])
        assert identity not in candidate_seen;candidate_seen.add(identity)
        original,metric=scores[identity]
        for k,value in metric.items():
            if k=='per_class':continue
            independent.compare(value,csv_value(row[k]),'/all_candidates/'+str(identity)+'/'+k)
        assert csv_value(row['selected_scopes'])==original['selected_scopes']
        if identity[2]=='audit' and 'family' in row:
            metadata=audit_meta[identity[0],identity[1],identity[5],identity[3]+'/'+identity[4],identity[6]]
            for field in ('family','candidate_origin','source_view','source_candidate_id','fit_rows','fit_support','fit_coverage_complete','space','reused_historical_fit','inherited_singleton'):
                independent.compare(metadata.get(field),csv_value(row[field]),'candidate_metadata/'+field)
        expected=priors[identity[0]][identity[4]][sk(identity[-2],identity[-1])]['log_loss']-metric['log_loss'] if identity[2]=='audit' else None
        assert_number(row['absolute_recovery'],expected,identity)
        report['candidate_score_rows']+=1
    assert candidate_seen==set(scores) and not independent.errors,independent.errors[:3]
    class_seen=set()
    for row in table('PER_CLASS'):
        identity=(int(row['seed']),row['condition'],row['role'],row['view'],row['target'],int(row['audit_budget']) if row['audit_budget'] else None,row['candidate_id'],row['split'],row['weight'])
        original,metric=scores[identity];category=int(row['class_index']);key=(*identity,category)
        assert key not in class_seen;class_seen.add(key)
        for field,value in metric['per_class'][category].items():
            independent.compare(value,csv_value(row[field]),'per_class/'+field)
        if 'fit_class_support' in row:
            metadata=audit_meta.get((identity[0],identity[1],identity[5],identity[3]+'/'+identity[4],identity[6]),{})
            support=metadata.get('fit_support')
            expected=support[category] if support is not None and category<len(support) else None
            assert_number(row['fit_class_support'],expected,'fit_class_support')
    assert len(class_seen)==sum(len(metric['per_class']) for original,metric in scores.values())
    report['per_class_rows']=len(class_seen)
    def point(row,condition_key='condition'):
        return points[int(row['seed']),row[condition_key],row['split'],row['weight'],int(row['budget']),row['scope']]
    flat_seen=set();groups={}
    for row in table('PER_SEED'):
        pt=point(row);h=point({**row,'condition':'H'});kind,ep=row['kind'],row['endpoint']
        value={'utility_loss':pt['utility'],'attack_loss':pt['attack'],'absolute_recovery':pt['gains'],
               'additional_recovery':{e:v-h['gains'][e] for e,v in pt['gains'].items()},
               'utility_gain_vs_H':{t:h['utility'][t]-v for t,v in pt['utility'].items()}}[kind][ep]
        assert_number(row['value'],value,row)
        identity=tuple(row[k] for k in ('condition','split','weight','budget','scope','kind','endpoint'))
        key=(int(row['seed']),*identity);assert key not in flat_seen;flat_seen.add(key)
        groups.setdefault(identity,[]).append((int(row['seed']),value));report['selected_endpoint_rows']+=1
    assert len(flat_seen)==len(points)*(5+11+11+11+5)
    for row in table('MEANS'):
        vals=groups[tuple(row[k] for k in ('condition','split','weight','budget','scope','kind','endpoint'))]
        assert_number(row['mean'],statistics.mean(v for s,v in vals),row)
        assert_number(row['sample_sd'],statistics.stdev(v for s,v in vals) if len(vals)>1 else None,row)
        assert int(row['n_seeds'])==len(vals);report['mean_rows']+=1
    def vector(a,b,tasks,delta,tolerance=1e-12):
        du={t:a['utility'][t]-b['utility'][t] for t in tasks}
        dg={e:a['gains'][e]-b['gains'][e] for e in sensitive}
        utility=all(v<=delta+1e-12 for v in du.values());privacy=all(v<=tolerance for v in dg.values())
        strict=any(v < -1e-12 for v in (*du.values(),*dg.values()))
        return {'assessable':True,'utility_no_worse':utility,'sensitive_no_worse':privacy,'strict_improvement':strict,
                'dominates':utility and privacy and strict,'utility_differences':du,'sensitive_differences':dg}
    def check_dict(actual,expected,context):
        for k,v in expected.items():independent.compare(v,csv_value(actual[k]) if isinstance(actual[k],str) else actual[k],context+'/'+k)
    def source_ok(pt,s,split,weight):
        return all(pt['utility'][t]<=parent[s,split,weight][t]+.01+1e-12 for t in source)
    pairs=(('spectral_C1','spectral_L1'),('spectral_C1','spectral_L2'),('spectral_C025','spectral_L025'))+tuple((c,r) for c in SPECTRAL for r in ('J','H','E','A0'))+tuple(('J',r) for r in ('H','E','A0','L025','L20'))
    pair_contexts={(s,left,right,split,w,b,scope) for left,right in pairs for s,split,w,b,scope in itertools.product(range(3),('validation','test'),weights,(120,360),SCOPES) if (s,left,split,w,b,scope) in points and (s,right,split,w,b,scope) in points}
    pair_seen=set();vector_seen=set();match_seen=set();pair_groups={}
    def pair_context(row):return (int(row['seed']),row['left'],row['right'],row['split'],row['weight'],int(row['budget']),row['scope'])
    for row in table('PAIRED'):
        identity=(*pair_context(row),row['kind'],row['endpoint']);assert identity not in pair_seen;pair_seen.add(identity)
        a,b=point(row,'left'),point(row,'right');value=a[row['kind']][row['endpoint']]-b[row['kind']][row['endpoint']]
        assert_number(row['difference'],value,row)
        identity=tuple(row[k] for k in ('left','right','split','weight','budget','scope','kind','endpoint'))
        pair_groups.setdefault(identity,[]).append(value);report['paired_rows']+=1
    assert pair_seen=={(*context,kind,ep) for context in pair_contexts for kind,endpoints in [('utility',TASKS),('gains',tuple(v+'/'+t for v,ts in ROLES.items() for t in ts))] for ep in endpoints}
    for row in table('PAIRED_AGGREGATE'):
        values=pair_groups[tuple(row[k] for k in ('left','right','split','weight','budget','scope','kind','endpoint'))]
        assert_number(row['mean'],statistics.mean(values),row)
        assert_number(row['sample_sd'],statistics.stdev(values) if len(values)>1 else None,row)
    for row in table('SOURCE_FEASIBILITY'):
        s=int(row['seed']);pt=points[s,row['condition'],row['split'],row['weight'],360,'kernel_expanded_catchup']
        t=row['task'];loss=pt['utility'][t];reference=parent[s,row['split'],row['weight']][t]
        check_dict(row,{'loss':loss,'parent_loss':reference,'loss_minus_parent':loss-reference,'excess_over_allowance':loss-reference-.01,
                       'pass':loss<=reference+.01+1e-12,'all_source_pass':source_ok(pt,s,row['split'],row['weight'])},'source_feasibility')
    for row in table('DIRECTIONAL_VECTORS'):
        identity=(*pair_context(row),row['panel'],float(row['delta']));assert identity not in vector_seen;vector_seen.add(identity)
        check_dict(row,vector(point(row,'left'),point(row,'right'),rules['panels'][row['panel']],float(row['delta'])),'directional')
        report['directional_rows']+=1
    assert vector_seen=={(*context,panel,delta) for context in pair_contexts for panel in rules['panels'] for delta in rules['delta_values']}
    for row in table('UTILITY_MATCHES'):
        identity=(*pair_context(row),row['panel'],float(row['delta']),row['attribute']);assert identity not in match_seen;match_seen.add(identity)
        a,b=point(row,'left'),point(row,'right');tasks=rules['panels'][row['panel']];delta=float(row['delta']);s=int(row['seed'])
        good=source_ok(a,s,row['split'],row['weight']) and source_ok(b,s,row['split'],row['weight'])
        close=all(abs(a['utility'][t]-b['utility'][t])<=delta+1e-12 for t in tasks)
        direction=all(a['utility'][t]-b['utility'][t]<=delta+1e-12 for t in tasks)
        dg=a['gains']['AB/'+row['attribute']]-b['gains']['AB/'+row['attribute']];strict=dg < -1e-12
        check_dict(row,{'both_source_feasible':good,'utility_close':close,'utility_directional':direction,'gain_difference':dg,
            'strict_gain_improvement':strict,'qualifies_close':good and close and strict,'qualifies_directional':good and direction and strict},'matching')
        report['matching_rows']+=1
    assert match_seen=={(*context,panel,delta,attr) for context in pair_contexts for panel in rules['panels'] for delta in rules['delta_values'] for attr in ('SEX','RAC1P')}
    # Reconstruct routed per-person arrays without importing reporting helpers.
    loss_cache={};archives={};label_cache={};raw_cache={};mixed={};uniforms={}
    def get_losses(s,c,record,split):
        role,v,t,b,cid=(record[k] for k in ('role','view','target','audit_budget','candidate_id'))
        key=f'{role}/{v}/{t}/{b}/{cid}/{split}';cachekey=(s,c,key)
        pool=('downstream_validation' if role=='utility' else 'attacker_validation') if split=='validation' else 'test'
        if s not in label_cache:
            lp=out/f'seed_{s}/evaluation_labels.npz';checked[str(lp)]=sha(lp);label_cache[s]=arrays(lp)
            raw_cache[s]=arrays(ancestor/f'seed_{s}/split_rows.npz')
        if cachekey not in loss_cache:
            if (s,c) not in archives:
                path=out/f'seed_{s}'/c/'predictions.npz';checked[str(path)]=sha(path);archives[s,c]=np.load(path)
            y=label_cache[s][pool+'/'+t];valid=y>=0;p=archives[s,c][key]
            clipped=np.maximum(1e-12,np.minimum(1.,p));clipped=clipped/clipped.sum(1)[:,None]
            loss_cache[cachekey]=-np.log(clipped[np.arange(valid.sum()),y[valid]])
        valid=label_cache[s][pool+'/'+t]>=0
        return loss_cache[cachekey],valid,pool
    for row in table('WITHHOLDING'):
        s,c,p=int(row['seed']),row['condition'],float(row['p']);a=point(row);h=point({**row,'condition':'H'})
        ep='utility/'+row['target'] if row['role']=='utility' else 'audit/'+row['view']+'/'+row['target']
        hr,ar=h['selected'][ep],a['selected'][ep]
        assert row['H_candidate']==hr['candidate_id'] and row['augmented_candidate']==ar['candidate_id']
        hl,valid,pool=get_losses(s,'H',hr,row['split']);al,_,_=get_losses(s,c,ar,row['split'])
        ukey=(s,c,pool)
        if ukey not in uniforms:
            uniforms[ukey]=np.asarray([int.from_bytes(hashlib.sha256(('spectral-withholding-v1|%d|%s|%d'%(s,c,int(raw))).encode()).digest()[:8],'big')/2**64 for raw in raw_cache[s][pool]])
        branches=uniforms[ukey][valid]<p;expected=hl*(1-p)+al*p;sampled=np.where(branches,al,hl)
        w=label_cache[s]['weights/'+pool][valid] if row['weight']=='person_weighted' else np.ones(len(hl))
        average=lambda v:float(np.dot(v,w)/w.sum())
        check_dict(row,{'expected_loss':average(expected),'sampled_loss':average(sampled),'H_loss':average(hl),'augmented_loss':average(al),
                       'sampled_augmented_fraction':average(branches),'valid_rows':int(valid.sum())},'withholding')
        mk=(s,row['mechanism'],row['split'],row['weight'],int(row['budget']),row['scope'])
        pt=mixed.setdefault(mk,{'utility':{},'gains':{}})
        if row['role']=='utility':pt['utility'][row['target']]=average(expected)
        else:
            target=row['target'];endpoint=row['view']+'/'+target
            prior=priors[s][target][sk(row['split'],row['weight'])]['log_loss']
            pt['gains'][endpoint]=prior-average(expected)
            assert_number(row['absolute_recovery'],pt['gains'][endpoint],row)
            assert_number(row['additional_recovery'],average(hl)-average(expected),row)
        report['withholding_rows']+=1
    for archive in archives.values():archive.close()
    assert report['withholding_rows']==len([1 for s,c in counts['available'] if c in ('E','A0','L025','L20')])*2*2*6*16*5*2
    for row in table('WITHHOLDING_COMPARISONS'):
        left=point(row,'left');right=mixed[int(row['seed']),row['right'],row['split'],row['weight'],int(row['budget']),row['scope']]
        check_dict(row,vector(left,right,rules['panels'][row['panel']],float(row['delta'])),'withholding_comparison')
        good=source_ok(left,int(row['seed']),row['split'],row['weight']) and source_ok(right,int(row['seed']),row['split'],row['weight'])
        assert csv_value(row['both_source_feasible'])==good
        if 'withholding_dominates' in row:
            reverse=vector(right,left,rules['panels'][row['panel']],float(row['delta']))
            assert csv_value(row['withholding_dominates'])==reverse['dominates']
            independent.compare(reverse,csv_value(row['withholding_vector']),'reverse_withholding_vector')
            close=all(abs(left['utility'][t]-right['utility'][t])<=float(row['delta'])+1e-12 for t in rules['panels'][row['panel']])
            assert csv_value(row['utility_close'])==close
        report['withholding_comparison_rows']+=1
    routing=read(out/'WITHHOLDING_VALIDATION.json')
    for row in routing['route_identity']:
        key=(row['seed'],row['condition'],row['pool']);u=uniforms[key]
        assert row['uniform_sha256']==hashlib.sha256(u.tobytes()).hexdigest()
        assert row['raw_rows_sha256']==hashlib.sha256(raw_cache[row['seed']][row['pool']].tobytes()).hexdigest()
    decision=read(out/'DECISION.json');checked[str(out/'DECISION.json')]=sha(out/'DECISION.json')
    # Full fixed gate, independently using scalar endpoint differences.
    if counts['complete']:
        qualifying=[]
        def gate(left,right):
            comparisons=[]
            for s,w in itertools.product(range(3),weights):
                a,b=points[s,left,'test',w,360,'kernel_expanded_catchup'],points[s,right,'test',w,360,'kernel_expanded_catchup']
                differences=vector(a,b,TASKS,.001,.0005+1e-12)
                supported=all(a['coverage'][e] and b['coverage'][e] for e in sensitive)
                comparisons.append((supported,differences))
            nonworse=all(d['utility_no_worse'] and d['sensitive_no_worse'] for ok,d in comparisons)
            strict=all(d['utility_differences']['same_residence']<-.001 for ok,d in comparisons) or any(all(d['sensitive_differences'][e]<-.001 for ok,d in comparisons) for e in sensitive)
            return nonworse and strict
        for c in SPECTRAL:
            source_pass=all(source_ok(points[s,c,split,w,360,'kernel_expanded_catchup'],s,split,w) for s,split,w in itertools.product(range(3),('validation','test'),weights))
            residence=all(points[s,'H','test',w,360,'kernel_expanded_catchup']['utility']['same_residence']-points[s,c,'test',w,360,'kernel_expanded_catchup']['utility']['same_residence']>=.01-1e-12 for s,w in itertools.product(range(3),weights))
            passed=source_pass and residence and gate(c,'J')
            assert decision['checks'][c]['qualifies']==passed
            if passed:qualifying.append(c)
        assert decision['qualifying_arms']==sorted(qualifying)
        assert decision['nominee']==(min(qualifying) if qualifying else None)
        assert decision['coordination_supported']==all(gate('spectral_C1',right) for right in ('spectral_L1','spectral_L2'))
    else:assert decision['nominee'] is None and decision['decision_status']=='incomplete_no_decision'
    assert not independent.errors,independent.errors[:3]
    for path,digest in checked.items():assert sha(path)==digest,('report input changed',path)
    report.update(passed=True,complete=counts['complete'],checked_artifacts=checked,runtime_seconds=time.perf_counter()-start,
                  nomination_verified=True,stochastic_per_person_routes_verified=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results'/NAME)
    p.add_argument('--historical-root',type=Path,default=Path('/Users/nathansamson/PCRL'))
    p.add_argument('--report',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--reporting-only',action='store_true');a=p.parse_args()
    if a.report.exists():raise FileExistsError('Use a fresh evidence path')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1),torch.inference_mode():
        if a.reporting_only:r=verify_reporting(a.out,a.historical_root,a.allow_incomplete)
        else:
            r=Replay(a.out,a.historical_root,a.allow_incomplete).run()
            if (a.out/'REPORT_COUNTS.json').exists():
                r['reporting']=verify_reporting(a.out,a.historical_root,a.allow_incomplete)
                r['reporting_artifacts_verified']=True
    a.report.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    if not a.reporting_only:a.report.with_suffix('.md').write_text(f"# Independent spectral replay\n\n{'Complete' if r['complete'] else 'Explicit partial calibration'}: {r['systems']}/42 systems; {r['prediction_arrays']} exact prediction arrays; {r['score_dictionaries']} independently recalculated score dictionaries; maximum discrepancy {r['max_score_absolute_error']:.3g}. All {r['files_unchanged']} checked files unchanged. Zero fits or optimizer updates.\n\nAggregate/stochastic/comparison tables verified: {r['reporting_artifacts_verified']}. This verifies frozen development fixtures, not independent population performance.\n")
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
