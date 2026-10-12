"""Independent replay of every saved FIT checkpoint and common nomination."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(a.spec).read_text())
    def checked(pin):
        assert_active_from_env(stage='verification')
        if sha(pin['path'])!=pin['sha256']:raise ValueError('FIT replay source hash mismatch')
        return Path(pin['path'])
    import numpy as np,torch,joblib
    from .science import fit_fold,encoder,fold_pipeline,softmax,own_release,metrics,normalized_task_violation,common_select
    from .checkpoint_io import read
    from .conditional_laws import specs,draw_selection,diagnostic,raw_diagnostic
    torch.set_num_threads(1)
    source=json.loads(checked(cfg['source_manifest']).read_text())
    fit=source['roles']['OSF_DEFENSE_FIT']
    def load(pin):
        with np.load(checked(pin),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    data=load(fit['features']);label_files=[load(fit[name]) for name in ('income','occupation','sex')]
    for f in label_files:
        if not all(np.array_equal(f[k],data[k]) for k in ('row_id','unit')):raise ValueError('FIT label person mismatch')
    yi,yo,sex=[f[name] for f,name in zip(label_files,('income','occupation','sex'))]
    folds=fit_fold(data['unit']);train=folds!=2;val=folds==2;x=data['X'][val];labels=[yi[val],yo[val]]
    train_specs=specs([yi[train],yo[train]],sex[train],data['unit'][train]);val_specs=specs(labels,sex[val],data['unit'][val])
    baselines={}
    for seed in (0,1):
        pins=source['seeds'][str(seed)]['pins'];state=torch.load(checked(pins['model']),map_location='cpu',weights_only=True)
        tasks=[]
        for r in (0,1):
            head=joblib.load(checked(pins[f'head{r}']));w,b,c=fold_pipeline(head)
            tasks.append(metrics(labels[r],softmax(encoder(state,r,x)@w.T+b)))
        baselines[seed]=tasks
    allrows=[];checked_rows=[];sequential_freezes=[];updatecounts=[]
    for pin in cfg['fit_results']:
        result=json.loads(checked(pin).read_text())
        if result['family'] not in cfg['family_strengths']:raise ValueError('unregistered family')
        first_prefix=None
        if result['family'] in ('SEQ12','SEQ21') and 'stages' in result:
            first_prefix={row['epoch']:row for row in result['stages'][0]['checkpoints']}
        for row in result['checkpoints']:
            checkpoint=checked({'path':row['checkpoint'],'sha256':row['checkpoint_sha256']})
            payload=read(checkpoint)
            if payload['seed']!=row['seed'] or payload['epoch']!=row['epoch'] or payload['strength']!=row['strength']:
                raise ValueError('FIT checkpoint metadata mismatch')
            out=own_release(payload['model'],x)
            tm=[metrics(labels[r],out[f'p{r+1}']) for r in (0,1)]
            diff=max(abs(tm[r][m]-row['metrics'][r][m]) for r in (0,1) for m in ('accuracy','nll','brier'))
            violation=normalized_task_violation(tm,baselines[row['seed']]);nll=sum(t['nll'] for t in tm)
            if diff>1e-10 or abs(violation-row['violation'])>1e-8 or abs(nll-row['nll'])>1e-10:raise ValueError('FIT checkpoint metric/violation replay differs')
            family=result['family']
            if family=='RAW_JOINT':sensitive=raw_diagnostic(payload,x,sex[val],row['seed'])
            elif family=='U_NEW':sensitive=0.
            else:
                # A sequential nominee is assessed under the complete actual
                # local+pair law, regardless of its last active optimizer stage.
                method='U_TASKPOWER' if family=='U_TASKPOWER' else family
                draws=draw_selection(train_specs,val_specs,method,row['seed'])
                sensitive=diagnostic(dict(payload,family=family),x,draws)['sensitive']
            if abs(sensitive-row['sensitive'])>1e-8:raise ValueError('FIT sensitive diagnostic independent replay differs')
            ownrow=dict(row,metrics=tm,violation=violation,nll=nll,sensitive=sensitive,family=family)
            allrows.append(ownrow)
            checked_rows.append({'family':family,'seed':row['seed'],'strength':row['strength'],'epoch':row['epoch'],
                'checkpoint_sha256':row['checkpoint_sha256'],'metric_max_abs_error':diff,'sensitive_error':abs(sensitive-row['sensitive'])})
            expected=int(row['epoch'])*math.ceil(int(train.sum())/256)
            if payload['updates_per_recipient']!=[expected,expected]:raise ValueError('nominee per-recipient update count differs')
            updatecounts.append({'family':family,'seed':row['seed'],'strength':row['strength'],'epoch':row['epoch'],'updates_per_recipient':[expected,expected]})
            if first_prefix is not None and row['epoch']:
                first=(0 if family=='SEQ12' else 1)
                frow=first_prefix[row['epoch']];first_state=read(checked({'path':frow['checkpoint'],'sha256':frow['checkpoint_sha256']}))['model']
                names=[k for k in first_state if k.startswith((f'enc.{first}.',f'head.{first}.'))]
                if any(not torch.equal(first_state[k],payload['model'][k]) for k in names):raise ValueError('frozen sequential encoder/head changed')
                sequential_freezes.append({'family':family,'seed':row['seed'],'strength':row['strength'],'epoch':row['epoch'],'first_encoder_and_head_exact':True})
    selections={}
    for family,strengths in cfg['family_strengths'].items():
        rows=[r for r in allrows if r['family']==family]
        selected,fallback=common_select(rows,strengths,[0,5,10,20],family in ('U_NEW','U_TASKPOWER'))
        selections[family]={'selected':selected,'fallback':fallback}
    if 'LOCAL_TWIN' in selections:
        joint=selections['JOINT_CONDITIONAL']['selected']
        twinrows=[r for r in allrows if r['family']=='LOCAL_TWIN' and r['strength']==joint['strength'] and r['epoch']==joint['epoch']]
        if len(twinrows)!=2:raise ValueError('exact same strength/epoch local twin missing')
        selections['LOCAL_TWIN']['selected']={'strength':joint['strength'],'epoch':joint['epoch'],
            'violation':max(r['violation'] for r in twinrows),'nll':sum(r['nll'] for r in twinrows),'sensitive':sum(r['sensitive'] for r in twinrows)/2,
            'forced_exact_joint_configuration':True}
    if any(n.startswith('task_conditioned_representation.') and n!='task_conditioned_representation.scheduler' for n in sys.modules):raise ImportError('candidate scientific modules imported')
    body={'schema':'TCR1-R6-independent-FIT-selection-v1','status':'PASS','guard':guard,
         'checkpoints_replayed':len(checked_rows),'checkpoint_checks':checked_rows,'sequential_freeze_checks':sequential_freezes,
         'updates':updatecounts,'independent_selections':selections,'INNER_task_labels_read':False,'INNER_SEX_read':False,
         'training_pair_coverage_definition':'>=32 unique duplicate groups per SEX arm; fixed optimization weights; missing mass retained'}
    output=Path(cfg['output']);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(output)

if __name__=='__main__':main()
