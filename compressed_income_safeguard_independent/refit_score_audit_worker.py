"""Post-seal descriptive audit of all 54 immutable independent refit targets."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def run(inputs):
    import numpy as np,joblib
    from .replay import reader_table,weighted_auc,close
    cfg=inputs.cfg;lock=inputs.seal()
    if lock.get('role_label_sha256',{}).get('INNER_SELECTION',{}).get('sex')!=cfg['truth']['sha256']:
        raise PermissionError('refit audit truth differs from committed reporting label')
    truth=inputs.arrays(cfg['truth'],{'row_id','unit','sex'});registry=inputs.json(cfg['target_registry'])
    targets={t['target_id']:t for t in registry['targets']};items=cfg['attempts']
    if registry['status']!='FROZEN' or len(targets)!=54 or len(items)!=54 or {a['target_id'] for a in items}!=set(targets):
        raise ValueError('all 54 already frozen actual attempts required')
    if (len(truth['unit']),len(np.unique(truth['unit'])))!=(2235,2234):raise ValueError('refit audit actual assessment people/groups')
    records=[];aliases={};architecture={'LR':0,'MLP':0,'HGB':0}
    for item in sorted(items,key=lambda a:a['target_id']):
        t=targets[item['target_id']];freeze=inputs.json(t['freeze'])
        if t['freeze']['sha256'] not in lock['reader_freeze_sha256']:raise PermissionError('refit target bank absent from committed freeze')
        report=inputs.json(item['attempt_receipt']);matches=[r for r in report['records'] if r['target_id']==t['target_id']]
        if len(matches)!=1:raise ValueError('unique completed independent attempt receipt')
        saved=matches[0]
        if report['status']!='PASS' or saved['model_sha256']!=item['model']['sha256'] or saved['score_sha256']!=item['score']['sha256']:
            raise ValueError('actual refit bytes differ from queued attempt receipt')
        if saved['orientation']!=t['orientation'] or saved['reader']!=t['reader'] or saved['view']!=t['view']:
            raise ValueError('post-freeze target or ATT orientation changed')
        att=inputs.arrays(t['att_release']);inner=inputs.arrays(t['inner_release'],{'row_id','unit','r1','r2'})
        scores=inputs.arrays(item['score'],{'row_id','unit','score'})
        if any(not np.array_equal(x[k],truth[k]) for x in (inner,scores) for k in ('row_id','unit')):
            raise ValueError('refit audit truth and score people differ')
        x=reader_table(att['r1'],att['r2'],t['reader'],t['view']);xi=reader_table(inner['r1'],inner['r2'],t['reader'],t['view'])
        model=joblib.load(inputs.path(item['model']))
        if set(model)!={'model','mean','scale'} or not np.array_equal(model['model'].classes_,[0,1]):raise ValueError('independent refit model schema')
        mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
        if mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('independent refit preprocessing schema')
        raw_att=model['model'].predict_proba((x-mean)/scale)[:,1]
        positive=float(weighted_auc(att['sex'][att['validation']],raw_att[att['validation']])[0]);sign=1 if positive>=.5 else -1
        if sign!=t['orientation'] or freeze['validation'][t['view']][t['reader']]['orientation']!=sign:
            raise AssertionError('frozen ATT orientation differs from actual independently fitted reader')
        close(positive,saved['ATT_positive_auc'],'refit ATT positive AUC');close(max(positive,1-positive),saved['ATT_oriented_auc'],'refit ATT oriented AUC')
        oriented_att=raw_att if sign==1 else 1-raw_att
        independent_inner=model['model'].predict_proba((xi-mean)/scale)[:,1]
        if sign==-1:independent_inner=1-independent_inner
        close(independent_inner,scores['score'],'stored refit score versus its actual model')
        runner_att=inputs.arrays(item['runner_att_scores'])[t['score_key']]
        runner_inner=inputs.arrays(item['runner_inner_scores'])[t['score_key']]
        close(oriented_att,runner_att,'actual selected ATT target curve');close(scores['score'],runner_inner,'actual selected INNER target curve')
        digest=hashlib.sha256()
        for value in (att['row_id'],att['unit'],att['sex'],att['train'],att['validation'],x,inner['row_id'],inner['unit'],xi):
            value=np.ascontiguousarray(value);digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        actual_input=digest.hexdigest()
        if actual_input!=saved['actual_target_input_sha256']:raise AssertionError('actual registered target input digest differs')
        identity=(actual_input,t['reader'],t['model_seed'],sign);prior=aliases.get(identity)
        if prior and not np.array_equal(prior['curve'],scores['score']):raise AssertionError('repeated exact-target independent curves differ')
        kind='MLP' if 'MLP' in t['reader'] else 'HGB' if 'HGB' in t['reader'] else 'LR';architecture[kind]+=1
        records.append({'target_id':t['target_id'],'arm_id':t['arm_id'],'seed':t['seed'],'reader':t['reader'],
            'requested_view':t['requested_view'],'actual_view':t['view'],'frozen_score_key':t['score_key'],'ATT_orientation':sign,
            'ATT_positive_auc':positive,'ATT_oriented_auc':max(positive,1-positive),
            'INNER_refit_auc':float(weighted_auc(truth['sex'],scores['score'])[0]),
            'INNER_frozen_target_auc':float(weighted_auc(truth['sex'],runner_inner)[0]),
            'maximum_ATT_curve_difference':float(abs(oriented_att-runner_att).max()),
            'maximum_INNER_curve_difference':float(abs(scores['score']-runner_inner).max()),
            'model_sha256':item['model']['sha256'],'score_sha256':item['score']['sha256'],'actual_target_input_sha256':actual_input,
            'repeated_exact_target_of':None if prior is None else prior['target_id'],'status':'PASS'})
        aliases.setdefault(identity,{'target_id':t['target_id'],'curve':scores['score']})
    fit_count=sum(inputs.json(p)['numerical_fit_count'] for p in cfg['attempt_receipts'])
    if fit_count!=54 or len(cfg['attempt_receipts'])!=18:raise AssertionError('exact54 actual independent refits required')
    return {'schema':'CIS1-independent-fresh-refit-score-audit-v1','status':'PASS','registered_attempts':54,
        'actual_reader_refits':fit_count,
        'unique_actual_targets':len(aliases),'architecture_allocations':architecture,'records':records,
        'target_registry_sha256':cfg['target_registry']['sha256'],'reporting_lock_sha256':cfg['reporting_lock']['sha256'],
        'scope':'descriptive audit of ATT-selected immutable refit curves; no new selection or inferential family',
        'new_fits':0,'INNER_orientation_changes':0,'scientific_runner_imports':False,'no_fresh_population_claim':True}


def main():
    from .replay import Inputs
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);out=Path(inputs.cfg['public_output']);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','refit_attempts':54,'unique_targets':result['unique_actual_targets']}))

if __name__=='__main__':main()
