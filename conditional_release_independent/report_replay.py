"""Independent INNER score-table lookup, every reader and engineering probe.

The global all-reader freeze is checked before any INNER SEX array opens.
"""
import argparse,hashlib,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import digest,admitted_arrays
from .arithmetic import integrated_auc,validate_group_pairing


def independent_shuffle(y,groups):
    salt='pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2'
    random=np.random.Generator(np.random.PCG64DXSM(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16],'big')))
    buckets={}
    for group in sorted(set(groups)):
        ix=np.flatnonzero(groups==group);buckets.setdefault(len(ix),[]).append(ix)
    output=np.empty_like(y)
    for size in sorted(buckets):
        rows=buckets[size];permutation=random.permutation(len(rows))
        for j,target in enumerate(rows):output[target]=y[rows[int(permutation[j])]]
    return output


def auc(score,mass,label):
    laws=[]
    for s,m in zip(score,mass,strict=True):
        law={}
        for level,weight in zip(s,m,strict=True):
            if weight>0:law[float(level)]=law.get(float(level),0.)+float(weight)
        laws.append(law)
    return integrated_auc(laws,label)


def read_all(entry):
    with ZipFile(entry['path']) as z:
        names={x[:-4] for x in z.namelist()}
    return admitted_arrays(entry,names)


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unregistered independent reporting input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered independent reporting config')
    cfg=json.loads(Path(config).read_text());check(cfg['reporting_lock']);lock=json.loads(Path(cfg['reporting_lock']['path']).read_text())
    if lock['status']!='ALL_READERS_FROZEN' or lock['K']!=552:raise PermissionError('global freeze absent')
    for e in lock['reader_freezes']:
        check(e)
        if json.loads(Path(e['path']).read_text())['status']!='FROZEN_COMPLETE':raise PermissionError('incomplete frozen bank')
    results=[];anchor=None
    for item in cfg['reports']:
        for name in ('bank','law','scores','frozen','report'):check(item[name])
        bank=json.loads(Path(item['bank']['path']).read_text());law=admitted_arrays(item['law'],{'row_id','unit','code_r1','code_r2','mass'})
        # INNER SEX only materializes here, after every frozen reader receipt.
        expected=read_all(item['scores']);frozen=read_all(item['frozen']);reported=json.loads(Path(item['report']['path']).read_text())
        if not all(np.array_equal(expected[k],law[k]) for k in ('row_id','unit','mass')):raise ValueError('reporting release law differs')
        validate_group_pairing(expected['unit'],expected['sex'])
        if anchor is None:anchor={k:expected[k].copy() for k in ('row_id','unit','sex')}
        if not all(np.array_equal(anchor[k],expected[k]) for k in anchor):raise ValueError('reporting people/labels unpaired')
        indexes={'r1':law['code_r1'],'r2':law['code_r2'],'pair':law['code_r1']*len(bank['recipients']['2']['probability'])+law['code_r2']}
        y=expected['sex'];shuffled=independent_shuffle(y,law['unit']);views={};gap=0.
        for view,ix in indexes.items():
            values={}
            for key,table in frozen.items():
                if not key.startswith(view+'__') or key.endswith(('__SHUFFLED','__PLANTED')):continue
                actual=table[ix]
                if key not in expected or not np.array_equal(actual,expected[key]):raise ValueError('per-person expected score law differs from actual complete wire lookup')
                name=key[len(view)+2:];value=auc(actual,law['mass'],y);values[name]=value
                gap=max(gap,abs(value-reported['views'][view]['reader_auc'][name]))
            null=auc(np.full(law['mass'].shape,.5),law['mass'],y)
            shuffle=auc(frozen[view+'__SHUFFLED'][ix],law['mass'],shuffled)
            planted=auc(frozen[view+'__PLANTED'][ix,y[:,None]],law['mass'],y)
            controls={'null_auc':null,'shuffled_independent_inner_auc':max(shuffle,1-shuffle),'planted_auc':planted}
            gates={'null_exact':null==.5,'shuffled_at_most_0.60':controls['shuffled_independent_inner_auc']<=.60,'planted_at_least_0.99':planted>=.99}
            if gates!=reported['views'][view]['control_gates']:raise ValueError('engineering probe gate differs')
            for key,value in controls.items():gap=max(gap,abs(value-reported['views'][view]['controls'][key]))
            envelope=max(values.values());gap=max(gap,abs(envelope-reported['views'][view]['envelope_auc']))
            views[view]={'reader_auc':values,'envelope_auc':envelope,'controls':controls,'gates':gates}
        monotone=views['pair']['envelope_auc']+1e-12>=max(views['r1']['envelope_auc'],views['r2']['envelope_auc'])
        passed=all(all(v['gates'].values()) for v in views.values())
        if gap>1e-12 or not monotone or passed!=reported['all_controls_pass']:raise ValueError('reported recovery/complete-view/control status mismatch')
        results.append({'arm_id':item['arm_id'],'seed':item['seed'],'maximum_numeric_gap':gap,'all_primary_score_laws':'EXACT_WIRE_LOOKUP','views':views,'pair_ignore_other_monotonicity':monotone,'engineering_controls_pass':passed})
    report={'schema':'CBR1-independent-complete-reader-report-replay-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'reader_refits':0,'INNER_SEX_access_after_all_reader_freeze':True,'scope':'Independent complete transmitted-wire score lookup, every fixed reader AUC, local aliases, coalition envelope, null/shuffled/planted probe recovery and reporting gate. Failed engineering controls remain failed; PASS here means correct replay, not an automatic scientific result.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','reports':len(results)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(parser.parse_args().config)
