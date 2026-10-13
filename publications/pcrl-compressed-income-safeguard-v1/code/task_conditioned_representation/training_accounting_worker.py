"""Queue-only reconciliation of actual FIT logs and logical forward counts.

Forward/preprocessing counts are structurally derived from pinned fixed code,
not instrumented runtime events. No models are refitted or evaluated here.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path


def run(spec_path):
    from .scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(spec_path).read_text())
    def checked(item):
        assert_active_from_env(stage='verification')
        p=Path(item['path'])
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=item['sha256']:raise PermissionError('accounting input pin differs')
        return p
    for item in cfg['code_pins']:checked(item)
    initial=json.loads(checked(cfg['initial_index']).read_text());protected=json.loads(checked(cfg['protection_index']).read_text())
    fit_entries=[u for u in initial['units'] if u['family'] in ('U_NEW','U_TASKPOWER')]+protected['units']
    if len(fit_entries)!=46:raise ValueError('complete 46-fit accounting required')
    coverage=json.loads(checked(cfg['coverage']).read_text())
    train_coverage=coverage['FIT_train_folds01']
    local=sum(r['supported'] for key in ('income','occupation') for r in train_coverage[key]['strata'])
    pair=sum(r['supported'] for r in train_coverage['pair']['strata'])
    target=sum(math.comb(sum(sum(r['distinct_groups_by_sex'])>=32 for r in train_coverage[key]['strata']),2) for key in ('income','occupation'))
    totals={k:0 for k in ('model_steps','recipient_updates','task_minibatch_rows','recipient_task_rows','encoder_forward_calls','head_forward_calls','critic_forward_calls','new_critic_optimizer_steps','saved_checkpoints','gradient_logs','normalization_reference_refreshes','diagonal_reference_fits','kernel_core_calls','kernel_normalization_calls','kernel_distance_matrix_calls','raw_normalization_calls','gradient_norm_calls','model_loss_backward_calls')}
    per_family={};units=[];gradient_values={k:[] for k in ('task_gradient_norm','sensitive_gradient_norm','target_gradient_norm','effective_ratio')}
    latent_min=[];latent_max=[];head_norms=[];unmeasured=[]
    for entry in fit_entries:
        result_path=Path(entry['private_output'])/'PRIVATE_FIT_RESULT.json';result=json.loads(result_path.read_text())
        checked({'path':str(result_path),'sha256':cfg['fit_result_sha256'][entry['unit_id']]})
        stages=result.get('stages') or [result];counts={k:0 for k in totals};recipient=[0,0]
        for stage in stages:
            S=stage['steps'];E=stage['recipe']['epochs'];C=len(stage['checkpoints']);A=len(stage['active']);N=stage['train_rows']
            if S!=E*math.ceil(N/stage['recipe']['batch_size']):raise ValueError('observed task update count differs')
            logs_path=Path(stage['checkpoints'][0]['checkpoint']).parent/'GRADIENT_SCALE_LOG.json';logs=json.loads(logs_path.read_text());L=len(logs)
            if L!=math.ceil(S/stage['recipe']['gradient_log_every']):raise ValueError('observed gradient-log schedule differs')
            for record in logs:
                for key in gradient_values:
                    if record[key] is not None:gradient_values[key].append(record[key])
                variances=[v for vector in record['latent_variances'] for v in vector]
                latent_min.append(min(variances));latent_max.append(max(variances));head_norms.extend(record['head_norms'])
            has_objective=stage['target_enabled'] or stage['coefficient']!=0
            raw=stage['family']=='RAW_JOINT';R=E+1 if has_objective else 0
            enc=A*S+2*C+2*L+(2*(S+C)+2*R if has_objective else 0)+(10*S if raw else 0)
            heads=A*S+2*C+(2*(S+C)+2*R if has_objective else 0)+(10*S if raw else 0)
            counts['encoder_forward_calls']+=enc;counts['head_forward_calls']+=heads
            counts['model_steps']+=S;counts['recipient_updates']+=A*S
            for r in stage['active']:recipient[r]+=S
            counts['task_minibatch_rows']+=E*N;counts['recipient_task_rows']+=A*E*N
            counts['saved_checkpoints']+=C;counts['gradient_logs']+=L;counts['gradient_norm_calls']+=3*L;counts['model_loss_backward_calls']+=S
            counts['normalization_reference_refreshes']+=R;counts['diagonal_reference_fits']+=(3 if raw else 2)*R
            if raw:
                counts['critic_forward_calls']+=54*S+9*C;counts['new_critic_optimizer_steps']+=45*S
                counts['raw_normalization_calls']+=3*(6*S+C)
            elif has_objective:
                if stage['family']=='U_TASKPOWER':core=target;recipient_factor=target
                elif stage['family']=='JOINT_UNCONDITIONAL':core=target+3;recipient_factor=target+4
                else:core=local+pair+target;recipient_factor=local+2*pair+target
                counts['kernel_core_calls']+=S*core;counts['kernel_normalization_calls']+=S*2*recipient_factor;counts['kernel_distance_matrix_calls']+=S*3*recipient_factor
                # Selection support may differ from optimization support; count
                # the actual saved diagnostic lists rather than extrapolate it.
                for row in stage['checkpoints']:
                    p=Path(row['checkpoint']).parent/f"epoch_{row['epoch']}_selection.json";diagnostics=json.loads(p.read_text())['diagnostics']
                    details=diagnostics['draws'];factor=sum(2 if d['kind']=='pair' else 1 for d in details)
                    counts['kernel_core_calls']+=len(details);counts['kernel_normalization_calls']+=2*factor;counts['kernel_distance_matrix_calls']+=3*factor
        # Worker-level initial useful task metrics are outside generic fit().
        counts['encoder_forward_calls']+=2;counts['head_forward_calls']+=2
        if result['family']=='RAW_JOINT':
            counts['encoder_forward_calls']+=2;counts['head_forward_calls']+=2
            counts['normalization_reference_refreshes']+=1;counts['diagonal_reference_fits']+=3
            if result['critic_optimizer_steps']!=720+counts['new_critic_optimizer_steps']:raise ValueError('RAW inherited/new critic accounting differs')
        family=per_family.setdefault(result['family'],{k:0 for k in totals})
        for key in totals:totals[key]+=counts[key];family[key]+=counts[key]
        units.append({'unit_id':entry['unit_id'],'family':result['family'],'seed':result['seed'],'strength':result['strength'],'stage_calls':len(stages),'observed_search_updates_per_recipient':recipient,'counts':counts,'RAW_inherited_warm_counter':720 if result['family']=='RAW_JOINT' else 0})
    if totals['model_steps']!=53600 or totals['recipient_updates']!=80800 or totals['new_critic_optimizer_steps']!=216000:raise ValueError('complete registered update accounting differs')
    if totals['encoder_forward_calls']!=242032 or totals['head_forward_calls']!=236672:raise ValueError('logical fitting forward reconciliation differs')
    calibration={'workers':6,'encoder_forward_calls':732,'head_forward_calls':732,'critic_forward_calls':1728,'unique_warm_critic_optimizer_steps':1440,'gradient_norm_calls':192,'diagonal_reference_fits':34,'scope':'structural fixed16draw+80warm/update loops; separate from fitted continuations'}
    for item in cfg['calibration_results']:
        cal=json.loads(checked(item).read_text())
        if cal['draws']!=16 or not cal['supported']:raise ValueError('normalization accounting scope differs')
        if cal['family']=='RAW_JOINT' and (cal['critic_warm_updates_per_critic']!=80 or cal['critic_warm_optimizer_steps']!=720):raise ValueError('unique warm accounting differs')
    def summary(values):return {'count':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)} if values else None
    report={'schema':'TCR1-actual-training-accounting-v1','status':'PASS','guard':guard,'source_code_pins':[{'path':Path(item['path']).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix(),'sha256':item['sha256']} for item in cfg['code_pins']],
            'fitted_configurations':46,'actual_stage_calls':sum(u['stage_calls'] for u in units),'nomination_checkpoints':184,
            'fitting_totals':totals,'per_family':per_family,'units':units,'calibration':calibration,
            'unique_total_RAW_critic_optimizer_steps':totals['new_critic_optimizer_steps']+1440,
            'gradient_and_scale_summary':{**{k:summary(v) for k,v in gradient_values.items()},'latent_coordinate_variance_minima':summary(latent_min),'latent_coordinate_variance_maxima':summary(latent_max),'head_weight_norms':summary(head_norms)},
            'forward_and_preprocessing_count_basis':'STRUCTURAL_DERIVATION_FROM_PINNED_CODE_AND_ACTUAL_LOGS_NOT_INSTRUMENTED',
            'training_stratum_core_counts':{'local':local,'pair':pair,'target':target},
            'unmeasured':['variable privacy draw union-row cardinalities','wall/CPU cost per logical forward or preprocessing call','interpreter/import/source hashing I/O work; queue charges and conservative coordinator light-charge cover this'],
            'excludes':['source admission engineering forwards','synthetic fixtures/timing','nomination/independent verification/deployment/attack forwards; their queue units accounted separately'],
            'INNER_task_labels_read':False,'INNER_SEX_read':False}
    output=Path(cfg['output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();result=run(a.spec)
    print(json.dumps({'status':result['status'],'model_steps':result['fitting_totals']['model_steps'],'recipient_updates':result['fitting_totals']['recipient_updates']}))
if __name__=='__main__':main()
