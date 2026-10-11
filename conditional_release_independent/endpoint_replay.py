"""Independent streaming replay of all prospectively registered endpoints."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy import sparse
from .reconstruction import digest,admitted_arrays
from .arithmetic import integrated_auc,simultaneous_interval,validate_group_pairing
from .inference import weighted_rank_replicates


def arrays_score_mass(scores,mass,sex,groups,group_order):
    """Independent action-index aggregation, preserving exact score ties."""
    validate_group_pairing(groups,sex)
    active=mass>0
    scores_=scores[active];row,action=np.nonzero(active)
    levels,inverse=np.unique(scores_,return_inverse=True)
    group_index={g:i for i,g in enumerate(group_order)}
    rowgroup=np.array([group_index[g] for g in groups],int)[row]
    matrices=[]
    for label in (1,0):
        choose=sex[row]==label
        matrices.append(sparse.coo_matrix((mass[row[choose],action[choose]],(rowgroup[choose],inverse[choose])),shape=(len(group_order),len(levels))).tocsr())
    return {'positive':matrices[0],'negative':matrices[1],'scores':levels,'group_order':group_order}


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={i['path']:i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    if registered.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered replay config')
    cfg=json.loads(Path(config).read_text())
    def pin(entry):
        if registered.get(entry['path'])!=entry['sha256'] or digest(entry['path'])!=entry['sha256']:raise PermissionError('independent endpoint input unqueued')
        return entry
    family=json.loads(Path(pin(cfg['endpoint_family'])['path']).read_text())
    K=family['K']
    if len(family['endpoints'])!=K or len({x['endpoint_id'] for x in family['endpoints']})!=K:raise ValueError('prospective multiplicity slots')
    freeze=json.loads(Path(pin(cfg['reporting_lock'])['path']).read_text())
    if freeze['status']!='ALL_READERS_FROZEN' or freeze['K']!=K:raise PermissionError('allreader freeze absent')
    for entry in freeze['reader_freezes']:pin(entry)
    bootstrap=admitted_arrays(pin(cfg['bootstrap_counts']),{'group_values','counts'})
    order=bootstrap['group_values'];counts=bootstrap['counts']
    expected=np.random.default_rng(2026101105).multinomial(len(order),np.full(len(order),1/len(order)),size=2048)
    if not np.array_equal(counts,expected):raise ValueError('bootstrap counts differ from frozen paired sampler')
    with Path(pin(cfg['primary_csv'])['path']).open() as f:reported={row['endpoint_id']:row for row in csv.DictReader(f)}
    if set(reported)!={x['endpoint_id'] for x in family['endpoints']}:raise ValueError('primary endpoint IDs differ')
    distributions={};utility={};anchor=None;reader_count=0;maximum_bootstrap_gap=0.
    for record in cfg['recovery_inputs']:
        item=pin(record['pin'])
        with np.load(item['path'],allow_pickle=False) as archive:
            labels=archive['sex'];groups=archive['unit'];rows=archive['row_id'];mass=archive['mass']
            if anchor is None:anchor=(rows.copy(),groups.copy(),labels.copy())
            if not all(np.array_equal(value,reference) for value,reference in zip((rows,groups,labels),anchor,strict=True)):raise ValueError('people/seeds/labels unpaired')
            if not np.array_equal(np.unique(groups),order):raise ValueError('bootstrap reportinggroups mismatch')
            for view in ('r1','r2','pair'):
                envelope=np.full(len(counts),-np.inf);point=-np.inf;points={}
                for name in archive.files:
                    if not name.startswith(view+'__'):continue
                    score=archive[name]
                    prepared=arrays_score_mass(score,mass,labels,groups,order)
                    replicates=weighted_rank_replicates(prepared,counts,16)
                    actual=float(weighted_rank_replicates(prepared,np.ones((1,len(order))),16)[0])
                    envelope=np.maximum(envelope,replicates);point=max(point,actual);points[name]=actual;reader_count+=1
                if not points:raise ValueError('required frozen reader absent')
                distributions[(record['arm_id'],record['seed'],view)]={'point':point,'bootstrap':envelope,'reader_points':points}
    for record in cfg['utility_inputs']:
        item=pin(record['pin'])
        with np.load(item['path'],allow_pickle=False) as archive:
            groups=archive['unit'];rows=archive['row_id']
            if anchor is None or not np.array_equal(groups,anchor[1]) or not np.array_equal(rows,anchor[0]):raise ValueError('utility/recovery people unpaired')
            gi=np.searchsorted(order,groups);nrows=np.bincount(gi,minlength=len(order))
            for key in archive.files:
                if key in ('row_id','unit'):continue
                values=archive[key];totals=np.bincount(gi,weights=values,minlength=len(order))
                utility[(record['arm_id'],record['seed'],key)]={'point':float(np.mean(values)),'bootstrap':counts@totals/(counts@nrows)}
    def recovery(arm,seed,view):
        if seed=='mean':
            a,b=(distributions.get((arm,i,view)) for i in (0,1))
            return None if a is None or b is None else {'point':(a['point']+b['point'])/2,'bootstrap':(a['bootstrap']+b['bootstrap'])/2}
        return distributions.get((arm,seed,view))
    checked=[];gaps=[]
    with np.load(pin(cfg['primary_bootstrap'])['path'],allow_pickle=False) as frozen_bootstrap:
        for item in family['endpoints']:
            endpoint=item['endpoint_id'];row=reported[endpoint];d=None
            if item['kind']=='utility':d=utility.get((item['arm_id'],item['seed'],item['array_key']))
            elif item['kind']=='recovery':
                a=recovery(item['minuend'],item['seed'],item['view']);b=recovery(item['subtrahend'],item['seed'],item['view'])
                if a is not None and b is not None:d={'point':a['point']-b['point'],'bootstrap':a['bootstrap']-b['bootstrap']}
            else:raise ValueError('unknown endpointkind')
            if d is None:
                if row['status']!='UNTESTED_DEPENDENCY' or endpoint in frozen_bootstrap.files:raise ValueError('missing slot relabeled or estimated')
                checked.append({'endpoint_id':endpoint,'status':'UNTESTED_DEPENDENCY'});continue
            interval=simultaneous_interval(d['point'],d['bootstrap'],K,len(order))
            gap=float(np.max(np.abs(d['bootstrap']-frozen_bootstrap[endpoint]),initial=0));maximum_bootstrap_gap=max(maximum_bootstrap_gap,gap)
            if gap>1e-12:raise ValueError('paired bootstrap endpoint mismatch')
            for key in ('point','lower','upper'):
                gaps.append(abs(interval[key]-float(row[key])))
            gaps.extend([abs(interval['sd']-float(row['bootstrap_sd'])),abs(interval['critical']-float(row['critical_value']))])
            if item['direction']=='upper_cap':status='PASS' if interval['upper']<=item['threshold'] else 'VIOLATION' if interval['lower']>item['threshold'] else 'UNRESOLVED'
            elif item['direction']=='lower_target':status='PASS' if interval['lower']>=item['threshold'] else 'SHORTFALL' if interval['upper']<item['threshold'] else 'UNRESOLVED'
            elif item['direction']=='lower_strict':status='PASS' if interval['lower']>item['threshold'] else 'SHORTFALL' if interval['upper']<item['threshold'] else 'UNRESOLVED'
            else:raise ValueError('unknown frozen boundarydirection')
            if row['status']!=status:raise ValueError('boundary label mismatch')
            checked.append({'endpoint_id':endpoint,'status':status})
    if max(gaps,default=0)>1e-12:raise ValueError('primary interval value mismatch')
    result={'schema':'CBR1-independent-primary-endpoint-replay-v1','status':'PASS','unit_id':active['unit_id'],'K':K,'slots_checked':len(checked),'estimable_slots':len(gaps)//5,'fixed_reader_score_functions_replayed':reader_count,'maximum_endpoint_arithmetic_gap':max(gaps,default=0),'maximum_bootstrap_replicate_gap':maximum_bootstrap_gap,'paired_bootstrap_seed_and_counts':'EXACT','replicates':2048,'df':len(order)-1,'reader_refits':0,'status_counts':{s:sum(x['status']==s for x in checked) for s in sorted({x['status'] for x in checked})},'scope':'Independent actual-law score-mass rank aggregation, fixed reader envelope, paired people/seed inference and all registered interval boundaries; fixed fitted readers/policies; approximate development inference.'}
    with Path(cfg['output']).open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','K':K,'maximum_endpoint_gap':result['maximum_endpoint_arithmetic_gap']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(parser.parse_args().config)
