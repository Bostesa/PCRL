"""Independent full admitted-role raw83 routing and U0 reference replay."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .raw_forward import frozen_raw_probabilities,route_fine
from .reconstruction import digest,admitted_arrays


def route(policy,p):
    fine=policy['fine'];classes=np.asarray(fine['cell_class']);centroids=np.asarray(fine['centroid'],float);tokens=np.asarray(policy['cell_token']);decisions=p.argmax(1);out=np.empty(len(p),int)
    for cls in np.unique(decisions):
        rows=np.flatnonzero(decisions==cls);allowed=np.flatnonzero(classes==cls)
        # The row entropy is common to every candidate cell. Minimize cross
        # entropy. Scalar KL is independently used for numerically close ties.
        distances=-p[rows]@np.log(centroids[allowed]).T
        best=distances.argmin(1);out[rows]=tokens[allowed[best]]
        ordered=np.partition(distances,1,axis=1) if len(allowed)>1 else None
        close=[] if ordered is None else np.flatnonzero(ordered[:,1]-ordered[:,0]<1e-12)
        if len(close):out[rows[close]]=route_fine(policy,p[rows[close]])
    return out


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={e['path']:e['sha256'] for e in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unregistered full raw replay input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered full raw replay config')
    cfg=json.loads(Path(config).read_text());check(cfg['manifest']);manifest=json.loads(Path(cfg['manifest']['path']).read_text());reports=[]
    def checkall(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:check(x)
            else:
                for value in x.values():checkall(value)
    checkall(manifest)
    for seed in (0,1):
        source=manifest['seeds'][str(seed)];policy=json.loads(Path(source['pins']['policy']['path']).read_text());bank=json.loads(Path(source['public_bank']['path']).read_text())
        for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
            inputs=admitted_arrays(manifest['roles'][role]['features'],{'row_id','unit','X','feature_names'});old=admitted_arrays(source['roles'][role]['release'],{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'});refs=admitted_arrays(manifest['roles'][role][f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
            if inputs['feature_names'].tolist()!=bank['feature_names'] or not all(np.array_equal(inputs[k],old[k]) and np.array_equal(inputs[k],refs[k]) for k in ('row_id','unit')):raise ValueError('full raw schema/people differs')
            scores=frozen_raw_probabilities(source['pins'],inputs['X']);gaps=[]
            for r,p in enumerate(scores,1):
                codes=route(policy[f'p{r}'],p);probs=np.asarray(bank['recipients'][str(r)]['probability'])[codes]
                if not np.array_equal(codes,old[f'code_r{r}']) or not np.array_equal(probs,old[f'prob_r{r}']) or not np.array_equal(p.argmax(1),old[f'decision_r{r}']) or not np.array_equal(p.argmax(1),refs[f'd_{r}']):raise ValueError('full raw codes/probabilities/decisions differ')
                gap=float(np.max(np.abs(p-refs[f'U0_{r}']),initial=0));gaps.append(gap)
                if gap>1e-12:raise ValueError('full raw U0 useful reference differs')
            reports.append({'seed':seed,'role':role,'rows':len(inputs['X']),'raw83_codes_probabilities_decisions':'EXACT','U0_maximum_gaps':gaps})
    report={'schema':'CBR1-independent-full-raw83-replay-v1','status':'PASS','unit_id':active['unit_id'],'role_reports':reports,'rows_replayed':sum(x['rows'] for x in reports),'scientific_labels_read':False,'Ucal_forward_claimed':False,'new_models_fitted':0,'scope':'Independent functional encoder/frozen third-party head forward on every admitted raw83 row, label-free fine-cell routing, H-CLASS decoded probabilities and decision identities, U0 original reference full-row equality. Ucal remains a pinned admitted predecessor reference, not newly fitted or independently recalibrated here.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','rows':report['rows_replayed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
