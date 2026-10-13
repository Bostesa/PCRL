"""Globally frozen paired-group bootstrap utility/recovery endpoint worker."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(spec_path):
    from .attack_worker import queue_config,reporting_lock,read_reporting_sex
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    from .attacks import READERS,OUTPUT_READERS,REP_READERS,write_json,save_npz
    from .inference import auc,bootstrap_auc,group_bootstrap,interval,task_vectors
    from .utility_worker import reference_probability_key
    reporting_lock(cfg,verify)
    registry=json.loads(verify(cfg['endpoint_registry']['path'],cfg['endpoint_registry']['sha256']).read_text())
    expected={'primary':51,'utility':240,'secondary':33}
    if registry.get('K')!=expected or any(len(registry.get('families',{}).get(k,[]))!=v for k,v in expected.items()):raise PermissionError('finite endpoint family and preserved multiplicity differ')
    up=verify(cfg['utility_inputs']['path'],cfg['utility_inputs']['sha256'])
    with np.load(up,allow_pickle=False) as z:utility={k:z[k].copy() for k in z.files}
    if any(reference_probability_key(s,r,i) not in utility for s in (0,1) for r in ('U0','Ucal') for i in (1,2)):raise PermissionError('distinct original reference probability keys absent')
    labels=read_reporting_sex(cfg['inner_sex'],verify,utility);sex=labels['sex'];unit=labels['unit'];G=len(np.unique(unit))
    weights=group_bootstrap(unit,2048,2026101208);denominator=weights.sum(1);envelopes={};absolute=[];admissions={};primary_available={}
    for item in cfg['instances']:
        admissions[item['arm_id']]=admissions.get(item['arm_id'],True) and bool(item.get('point_eligible',False))
        primary_available[item['arm_id']]=primary_available.get(item['arm_id'],True) and bool(item.get('is_nominee_or_exact_alias',False))
        if item.get('contract','new_continuous')=='legacy_complete_score_wire':
            from conditional_release.inference import envelope_distribution
            from conditional_release.attacks import law_arrays
            from .source import read_exact
            bank=json.loads(verify(item['bank']['path'],item['bank']['sha256']).read_text())
            verify(item['inner_law']['path'],item['inner_law']['sha256']);verify(item['score_tables']['path'],item['score_tables']['sha256'])
            law=read_exact(item['inner_law'],{'row_id','unit','code_r1','code_r2','mass'})
            if not all(np.array_equal(law[k],labels[k]) for k in ('row_id','unit')):raise PermissionError('legacy secondary people are not paired')
            _,_,indexes,mass=law_arrays(law,bank)
            with np.load(item['score_tables']['path'],allow_pickle=False) as z:tables={k:z[k].copy() for k in z.files}
            group_values,first=np.unique(unit,return_index=True);counts=weights[:,first]
            for view in ('r1','r2','pair'):
                prefix=view+'__';scientific={k[len(prefix):]:v[indexes[view]] for k,v in tables.items() if k.startswith(prefix) and k[len(prefix):] not in ('SHUFFLED','PLANTED')}
                distribution=envelope_distribution(scientific,mass,sex,unit,counts=counts,group_values=group_values)
                point,boots=distribution['point'],distribution['bootstrap'];envelopes[(item['arm_id'],item['seed'],'output',view)]=(point,boots)
                absolute.append({'arm_id':item['arm_id'],'seed':item['seed'],'contract':'legacy_complete_score_wire_codes_probabilities_decisions_metadata','view':view,'interval':interval(point,boots,1,G),'interval_role':'descriptive pointwise95%; different historical complete wire, not equal capacity new output contract'})
            continue
        release_path=verify(item['release']['path'],item['release']['sha256']);score_path=verify(item['scores']['path'],item['scores']['sha256'])
        with np.load(release_path,allow_pickle=False) as z:
            if not all(np.array_equal(z[k],labels[k]) for k in ('row_id','unit')):raise PermissionError('bootstrap arm/seed people are not paired')
        with np.load(score_path,allow_pickle=False) as z:scores={k:z[k].copy() for k in z.files}
        if item.get('contract','new_continuous')!='new_continuous':raise PermissionError('historical finite laws require separate secondary adapter')
        for view in ('r1','r2','pair'):
            prefix=view+'__';science={k[len(prefix):]:s for k,s in scores.items() if k.startswith(prefix) and not k[len(prefix):].startswith(('PLANTED','SHUFFLED'))}
            for scope,slate in (('full',READERS),('output',OUTPUT_READERS),('representation',REP_READERS)):
                selected={n:s for n,s in science.items() if n.split('__')[-1] in slate}
                if not selected:raise PermissionError('declared envelope slate empty')
                key=(item['arm_id'],item['seed'],scope,view);point=max(auc(sex,s) for s in selected.values());boots=np.full(2048,-np.inf)
                for s in selected.values():
                    for start in range(0,2048,32):boots[start:start+32]=np.maximum(boots[start:start+32],bootstrap_auc(sex,s,weights[start:start+32]))
                envelopes[key]=(point,boots)
                ci=interval(point,boots,1,G)
                absolute.append({'arm_id':item['arm_id'],'seed':item['seed'],'contract':scope,'view':view,'interval':ci,'interval_role':'descriptive pointwise 95%; not simultaneous conjunction'})
    # Predeclared descriptive seed means retain paired people and each seed's
    # independently fitted fixed-reader envelope inside every bootstrap draw.
    summaries=sorted({(arm,scope,view) for arm,seed,scope,view in envelopes})
    legacy_arms={i['arm_id'] for i in cfg['instances'] if i.get('contract')=='legacy_complete_score_wire'}
    for arm,scope,view in summaries:
        a=envelopes.get((arm,0,scope,view));b=envelopes.get((arm,1,scope,view))
        if a is None or b is None:continue
        point=(a[0]+b[0])/2;boots=(a[1]+b[1])/2
        absolute.append({'arm_id':arm,'seed':None,'aggregation':'paired_seed_mean','contract':'legacy_complete_score_wire_codes_probabilities_decisions_metadata' if arm in legacy_arms else scope,'view':view,'interval':interval(point,boots,1,G),'interval_role':'descriptive paired-seed mean pointwise95%; not simultaneous conjunction; seeds are fitted runs on same people'})
    results={};boot_arrays={};utility_cache={}
    def utility_endpoint(e):
        arm=e['arm'];seed=e['seed'];task=e['task'];ref=e['reference'];metric=e['metric'];i={'income':1,'occupation':2}[task]
        pkey=f's{seed}__{arm}__p{i}';rkey=reference_probability_key(seed,ref,i)
        if pkey not in utility or rkey not in utility:return None
        key=(arm,seed,task,ref,metric)
        if key not in utility_cache:
            cand=task_vectors(utility[task],utility[pkey]);reference=task_vectors(utility[task],utility[rkey]);base=metric.split('_')[0]
            vector=reference[base]-cand[base] if metric=='accuracy_loss' else cand[base]-reference[base]
            utility_cache[key]=(float(vector.mean()),(weights@vector)/denominator)
        return utility_cache[key]
    def privacy_endpoint(e,scope,require_primary):
        arm=e['arm'];control=e['comparator'];metric=e['metric'];view='pair' if metric=='pair_benefit' else 'r1' if metric=='r1_worsening' else 'r2'
        if require_primary and (not primary_available.get(arm,False) or not primary_available.get(control,False)):return None
        values=[]
        for seed in (0,1):
            a=envelopes.get((arm,seed,scope,view));b=envelopes.get((control,seed,scope,view))
            if a is None or b is None:return None
            sign=-1 if metric=='pair_benefit' else 1
            values.append((sign*(a[0]-b[0]),sign*(a[1]-b[1])))
        return (sum(v[0] for v in values)/2,sum(v[1] for v in values)/2)
    for family,entries in registry['families'].items():
        results[family]=[]
        for e in entries:
            quantity=utility_endpoint(e) if e['kind']=='utility' else privacy_endpoint(e,'output' if family=='secondary' else 'full',family=='primary')
            if quantity is None:
                results[family].append({**e,'status':'MISSING_DEPENDENCY','interval':None});continue
            point,boots=quantity;ci=interval(point,boots,expected[family],G);boot_arrays[e['id']]=boots
            if 'cap' in e:status='PASS' if ci['upper']<=e['cap'] else 'VIOLATION' if ci['lower']>e['cap'] else 'UNRESOLVED'
            elif 'target' in e:status='PASS' if ci['lower']>=e['target'] else 'SHORTFALL' if ci['upper']<e['target'] else 'UNRESOLVED'
            else:status='ESTIMATED'
            admitted=admissions.get(e['arm'],False) and (e['kind']=='utility' or admissions.get(e['comparator'],False))
            results[family].append({**e,'status':status,'interval':ci,'point_eligible_comparison':admitted,'diagnostic_only':not admitted})
    private=Path(cfg['private_output'])
    if '.private' not in private.resolve().parts or private.exists():raise PermissionError('fresh private inference output required')
    private.mkdir(parents=True,mode=0o700);boot_sha=save_npz(private/'BOOTSTRAP_ENDPOINTS.npz',**boot_arrays);weight_sha=save_npz(private/'BOOTSTRAP_ROW_WEIGHTS.npz',row_id=labels['row_id'],unit=unit,weights=weights)
    out={'schema':'TCR1-paired-group-conditional-inference-v2','reference_namespace':'REF_U0/REF_Ucal original admitted service probabilities','unit_id':active['unit_id'],'endpoint_registry_sha256':cfg['endpoint_registry']['sha256'],'K':expected,'groups':G,'rows':len(unit),'replicates':2048,'bootstrap_seed':2026101208,'bootstrap_endpoint_sha256':boot_sha,'bootstrap_weight_sha256':weight_sha,'families':results,'absolute_recovery':absolute,'fixed_models_readers':True,'two_seeds':'paired fitted runs on same people; averaged within each replicate','coverage':'approximate conditional development, exploratory on reused Adult people'}
    write_json(private/'INFERENCE_COMPLETE.json',out);write_json(cfg['public_output'],out,False);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec);print(json.dumps({'status':'REPORTED','families':{k:len(v) for k,v in out['families'].items()}}))
if __name__=='__main__':main()
