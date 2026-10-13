"""Queue-only FIT nomination draft; R00 finalizes locks after R6 replay.

This worker never reads INNER arrays. It applies the existing locked selector
and proves source aliases from encoder/head weights, without fresh fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path

UTILITY=('U_NEW','U_TASKPOWER')
PROTECTED=('JOINT_CONDITIONAL','LOCAL_TWIN','LOCAL_CONDITIONAL','JOINT_UNCONDITIONAL','RAW_JOINT','SEQ12','SEQ21')
FAMILIES=UTILITY+PROTECTED


def digest(path):
    p=Path(path)
    if p.is_symlink():raise PermissionError('symlink refused')
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(config_path):
    from .scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(config_path).read_text())
    def checked(item):
        assert_active_from_env(stage='verification')
        if digest(item['path'])!=item['sha256']:raise PermissionError('nomination input hash differs')
        return Path(item['path'])
    science=json.loads(checked(cfg['science_lock']).read_text())
    manifest=json.loads(checked(cfg['manifest']).read_text())
    if cfg['manifest']['sha256']!=science['source_manifest_sha256']:raise PermissionError('science source identity differs')
    private=Path(cfg['private_output'])
    if '.private' not in private.resolve().parts or private.exists():raise PermissionError('fresh private nomination output required')
    import torch
    torch.set_num_threads(1)
    from .trainer import select_common,atomic_json
    from .models import load_checkpoint
    from .useful_service import load_useful_model
    strengths={f:([0.] if f in UTILITY else list(science['strengths'])) for f in FAMILIES}
    if cfg.get('family_strengths',strengths)!=strengths:raise ValueError('registered family ladder differs')
    expected={(f,s,float(g)) for f in FAMILIES for s in (0,1) for g in strengths[f]}
    seen=set();rows=[];result_pins=[];stage_accounting=[]
    for item in cfg['fit_results']:
        result=json.loads(checked(item).read_text());identity=(result['family'],int(result['seed']),float(result['strength']))
        if identity not in expected or identity in seen:raise ValueError('duplicate or unregistered continuation')
        seen.add(identity);result_pins.append(item)
        if sorted(r['epoch'] for r in result['checkpoints'])!=science['recipe']['checkpoints']:raise ValueError('checkpoint opportunity differs')
        for row in result['checkpoints']:
            if (row['family'],int(row['seed']),float(row['strength']))!=identity:raise ValueError('candidate/result identity differs')
            if any(not math.isfinite(float(row[k])) for k in ('violation','nll','sensitive')):raise ValueError('nonfinite candidate diagnostic')
            checked({'path':row['checkpoint'],'sha256':row['checkpoint_sha256']})
            expected_updates=int(row['epoch'])*40
            if row['updates_per_recipient']!=[expected_updates,expected_updates]:raise ValueError('nominee updates are not matched per encoder')
            if result['family'] in ('SEQ12','SEQ21') and (row.get('selection_active')!=[0,1] or row.get('selection_family')!=result['family']):raise ValueError('sequential nominee diagnostic is incomplete')
            rows.append(row)
        stage_accounting.append({'family':result['family'],'seed':result['seed'],'strength':result['strength'],'updates_per_recipient':result['updates_per_recipient'],'staged_continuation_calls':len(result.get('stages',[])) or 1,'critic_optimizer_steps':result.get('critic_optimizer_steps')})
    if seen!=expected:raise ValueError('complete registered FIT comparison missing continuations')
    selections={}
    for family in FAMILIES:
        forced=(selections['JOINT_CONDITIONAL']['strength'],selections['JOINT_CONDITIONAL']['epoch']) if family=='LOCAL_TWIN' else None
        selections[family]=select_common(rows,family,tuple(strengths[family]),forced=forced)
    source_models={s:load_useful_model(manifest,s) for s in (0,1)}
    model_cache={}
    def instance(row,arm,kind):
        cache_key=row['checkpoint_sha256']
        if cache_key not in model_cache:model_cache[cache_key]=load_checkpoint(row['checkpoint'],cache_key)
        model,payload=model_cache[cache_key]
        if payload['seed']!=row['seed'] or payload['epoch']!=row['epoch'] or payload['strength']!=row['strength']:raise ValueError('checkpoint identity differs')
        if payload['updates_per_recipient']!=row['updates_per_recipient']:raise ValueError('checkpoint update accounting differs')
        state=model.state_dict();source=source_models[row['seed']].state_dict()
        exact_source=set(state)==set(source) and all(torch.equal(state[k],source[k]) for k in state)
        return {'arm_id':arm,'family':row['family'],'seed':row['seed'],'strength':row['strength'],'epoch':row['epoch'],
                'checkpoint':row['checkpoint'],'checkpoint_sha256':cache_key,'kind':kind,
                'exact_source_alias':exact_source,'checkpoint_training_family':payload['family'],'updates_per_recipient':row['updates_per_recipient']}
    references=[]
    for seed in (0,1):
        source_row=next(r for r in rows if r['family']=='U_NEW' and r['seed']==seed and r['epoch']==0)
        entry=instance(source_row,'U0','restored_zero_update_reference')
        if not entry['exact_source_alias']:raise ValueError('zero-update source reference differs')
        references.append(entry)
    nominees=[];fallbacks=[];public_families={}
    for family in FAMILIES:
        selection=selections[family]
        nominees.extend(instance(row,family,'FIT_nominee') for row in selection['nominees'])
        fallbacks.extend(instance(row,family+'_DIAGNOSTIC_FALLBACK','FIT_frozen_diagnostic_fallback') for row in selection['fallback']['nominees'])
        public_families[family]={'strength':selection['strength'],'epoch':selection['epoch'],'FIT_surrogate_feasible':selection['surrogate_feasible'],
            'worst_normalized_FIT_task_deviation':selection['violation'],'paired_seed_sum_NLL':selection['nll'],
            'paired_seed_sum_sensitive':selection['sensitive'],'paired_seed_mean_sensitive':selection['sensitive']/2,
            'fallback_strength':selection['fallback']['strength'],'fallback_epoch':selection['fallback']['epoch'],
            'nominee_checkpoint_sha256':[row['checkpoint_sha256'] for row in selection['nominees']],
            'fallback_checkpoint_sha256':[row['checkpoint_sha256'] for row in selection['fallback']['nominees']],
            'forced_exact_joint_configuration':family=='LOCAL_TWIN'}
    instances=references+nominees+fallbacks
    # Draft status cannot satisfy source.py's INNER task-label guard. R00 uses
    # the independently verified draft to create the separate final FROZEN lock.
    output={'schema':'TCR1-FIT-nomination-draft-v1','status':'FIT_SELECTED_AWAITING_R6','science_lock_sha256':cfg['science_lock']['sha256'],
            'source_manifest_sha256':cfg['manifest']['sha256'],'guard':guard,'nominees':references+nominees,'fallbacks':fallbacks,
            'instances':instances,'selections':selections,'families':public_families,'fit_result_pins':result_pins,
            'training_accounting':stage_accounting,'continuations':len(seen),'saved_candidates':len(rows),
            'INNER_task_labels_read':False,'INNER_SEX_read':False,'original_point_eligibility':'NOT_EVALUATED'}
    atomic_json(private,output);os.chmod(private,0o600)
    public={'schema':output['schema'],'status':output['status'],'science_lock_sha256':output['science_lock_sha256'],
            'source_manifest_sha256':output['source_manifest_sha256'],'private_draft_sha256':digest(private),'families':public_families,
            'instances':[{k:v for k,v in item.items() if k!='checkpoint'} for item in instances],
            'training_accounting':stage_accounting,'continuations':len(seen),'saved_candidates':len(rows),
            'INNER_task_labels_read':False,'INNER_SEX_read':False,'original_point_eligibility':'NOT_EVALUATED'}
    atomic_json(cfg['public_output'],public)
    return public


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec)
    print(json.dumps({'status':out['status'],'continuations':out['continuations'],'saved_candidates':out['saved_candidates']}))
if __name__=='__main__':main()
