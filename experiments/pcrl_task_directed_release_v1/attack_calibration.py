"""Predetermined attack-strength diagnostics from accepted JSON aggregates.

H, J, continuous_task and Trisk_code are fixed before reading outcomes. This
module fits/selects no deployment model and opens no NPZ or model artifacts.
Its positive-control observations are validation diagnostics, not inference.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import re

from . import config,run

CONTROLS=('H','J','continuous_task','Trisk_code')
ROLES=tuple('attack:'+r for r in config.PRIMARY)
OWN=('logistic','hist_gb_20','hist_gb_5','sampled_hist_gb_20','sampled_hist_gb_5','mlp_120','mlp_360')
WEIGHTS=('unweighted','weighted','balanced')
CONTINUATION='actual last weights and Adam/RNG state; no selected-state rewind'


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Invalid accepted JSON hash')
    return value


def _scores(values):
    if set(values)!=set(WEIGHTS):raise ValueError('Expected both weightings and balanced validation CE')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Invalid candidate validation CE')
    if not math.isclose(values['balanced'],.5*(values['unweighted']+values['weighted']),rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError('Inconsistent balanced validation CE')
    return {w:float(values[w]) for w in WEIGHTS}


def _unit(name,anchor,role):
    if name not in CONTROLS or isinstance(anchor,bool) or anchor not in (0,1,2) or role not in ROLES:
        raise ValueError('Unregistered calibration control/anchor/role')


def _best(scores,ids):return min(ids,key=lambda c:(scores[c]['balanced'],c))


def _gain(scores,baseline,candidate):
    return {w:scores[baseline][w]-scores[candidate][w] for w in WEIGHTS}


def role_record(name,anchor,role,native,metadata,*,receipt_sha256,selection_sha256,metadata_hashes):
    """Allowlist one accepted role's scores; omit private paths and row data."""
    _unit(name,anchor,role)
    view,target=role.split(':',1)[1].split('/')
    if (native.get('view'),native.get('target'),native.get('kind'))!=(view,target,'attack'):
        raise ValueError('Role selection record identity differs')
    candidates=native['candidates'];scores={c:_scores(s) for c,s in native['validation_scores'].items()}
    if set(candidates)!=set(scores) or not set(OWN)<=set(scores):raise ValueError('Incomplete registered attack candidate slate')
    own={c for c,s in candidates.items() if s.get('origin')=='independent'}
    if own!=set(OWN):raise ValueError('Independent attack slate differs from registration')
    public={}
    for cid,candidate in candidates.items():
        if not re.fullmatch('[A-Za-z0-9_.-]+',cid):raise ValueError('Invalid public candidate recipe ID')
        route=candidate['route']
        if (route.get('kind')!='model' or route.get('source_view') not in ('A','B','AB')
                or route.get('wire') not in ('H','release') or candidate.get('origin') not in ('independent','H_baseline','A_ancestor','B_ancestor')):
            raise ValueError('Unknown primary attack route')
        if cid in own and (route['source_view']!=view or route['wire']!=('H' if name=='H' else 'release')):
            raise ValueError('Independent control attacker has the wrong wire')
        public[cid]={'ce':scores[cid],'origin':candidate['origin'],
                     'route':{k:route[k] for k in ('kind','source_view','wire')}}
    selected,independent=native['selection'],native['independent_selection']
    if selected!=_best(scores,scores) or independent!=_best(scores,own):raise ValueError('Frozen validation winner differs from candidate CE/tie rule')
    if set(metadata)!={120,360}:raise ValueError('Both nested MLP checkpoint metadata are required')
    short,long=metadata[120],metadata[360]
    for epoch,meta in metadata.items():
        if (meta.get('candidate_id')!=f'mlp_{epoch}' or meta.get('parameters',{}).get('epochs')!=epoch
                or meta.get('trajectory_epochs')!=360 or meta.get('trajectory_continuation')!=CONTINUATION
                or not isinstance(meta.get('selected_epoch'),int) or not 0<=meta['selected_epoch']<=epoch):
            raise ValueError('Invalid nested MLP trajectory metadata')
        _hash(meta['initial_state_hash'])
    if any(short[k]!=long[k] for k in ('seed','schedule_seed','initial_state_hash')):
        raise ValueError('MLP360 is not the declared continuation from the same fresh initialization')
    standard=_best(scores,[c for c in OWN if c!='mlp_360'])
    tree=_best(scores,[c for c in OWN if 'hist_gb' in c])
    groups={'selected_deployment':selected,'independent_catchup':independent,'standard_without360':standard,
            'mlp120':'mlp_120','continued_mlp360':'mlp_360','best_tree':tree,'logistic':'logistic'}
    for k in ('fit_rows','validation_rows'):
        if isinstance(native[k],bool) or not isinstance(native[k],int) or native[k]<=0:raise ValueError('Invalid aggregate row count')
    return {'configuration':name,'anchor':anchor,'role':role,'selection':selected,'independent_selection':independent,
            'candidate_validation':public,'comparison_predictors':groups,
            'comparison_ce':{g:copy.deepcopy(scores[c]) for g,c in groups.items()},
            'continued360_gain_over120':_gain(scores,'mlp_120','mlp_360'),
            'catchup_gain_over_standard':_gain(scores,standard,independent),
            'best_tree_gain_over_logistic':_gain(scores,'logistic',tree),
            'nonlinear_gain_over_logistic':_gain(scores,'logistic',independent),
            'fit_rows':native['fit_rows'],'validation_rows':native['validation_rows'],
            'trajectory':{'same_initialization_and_schedule':True,'seed':short['seed'],'schedule_seed':short['schedule_seed'],
                          'initial_state_hash':short['initial_state_hash'],'selected_epoch120':short['selected_epoch'],
                          'selected_epoch360':long['selected_epoch'],'continuation':CONTINUATION,
                          'interpretation':'nested best-validation checkpoints through epochs120/360 of one uninterrupted trajectory; epoch120 begins from fresh initialization'},
            'provenance':{'receipt_sha256':_hash(receipt_sha256),'selection_sha256':_hash(selection_sha256),
                          'mlp_metadata_sha256':{str(e):_hash(metadata_hashes[str(e)]) for e in (120,360)}}}


def _checked_json(base,relative,receipt):
    path=base/relative;expected=receipt.get('artifact_hashes',{}).get(relative)
    if expected is None or run.sha(path)!=expected:raise ValueError('Accepted calibration JSON hash mismatch')
    return json.loads(path.read_text()),expected


def collect_calibration(*,out_root=config.OUT):
    """Explicit outcome-read entry point; only receipt-hashed JSON is opened."""
    root=Path(out_root);plan=root/'RESOURCE_SCHEDULE.json';plan_sha=run.sha(plan);schedule=json.loads(plan.read_text())
    cfg_hash=config.digest(config.configuration())
    if schedule.get('frozen_before_comparative_outcomes') is not True or schedule.get('config_hash')!=cfg_hash:
        raise ValueError('Attack calibration requires the prospectively frozen resource plan')
    rows=[];incomplete=[]
    for name in CONTROLS:
        for anchor in (0,1,2):
            base=root/'private/run'/f'anchor_{anchor}'/'audits'/name;marker=base/'COMPLETE.json'
            if not marker.exists():incomplete.append({'configuration':name,'anchor':anchor});continue
            receipt=json.loads(marker.read_text());run._validate_provenance(receipt,'audit')
            if receipt.get('anchor')!=anchor or receipt.get('configuration')!=name:raise ValueError('Accepted control identity differs')
            receipt_sha=run.sha(marker)
            for role in ROLES:
                safe=role.replace(':','__').replace('/','__')
                native,selection_sha=_checked_json(base,safe+'/selection.json',receipt)
                metadata={};hashes={}
                for epoch in (120,360):
                    metadata[epoch],hashes[str(epoch)]=_checked_json(base,f'{safe}/models/mlp_{epoch}/metadata.json',receipt)
                rows.append(role_record(name,anchor,role,native,metadata,receipt_sha256=receipt_sha,
                                        selection_sha256=selection_sha,metadata_hashes=hashes))
            if run.sha(marker)!=receipt_sha:raise ValueError('Accepted audit changed while collecting calibration')
    if run.sha(plan)!=plan_sha:raise ValueError('Resource schedule changed during calibration')
    return {'phase':'validation','evaluation_opened':False,'config_hash':cfg_hash,
            'source_hashes':run.source_fingerprint('audit'),'resource_schedule_sha256':plan_sha,
            'records':rows,'incomplete_units':incomplete}


def _public_record(row):
    fields={'configuration','anchor','role','selection','independent_selection','candidate_validation',
            'comparison_predictors','comparison_ce','continued360_gain_over120','catchup_gain_over_standard',
            'best_tree_gain_over_logistic','nonlinear_gain_over_logistic','fit_rows','validation_rows','trajectory','provenance'}
    if set(row)!=fields:raise ValueError('Unexpected or missing public calibration record field')
    scores={}
    for cid,candidate in row['candidate_validation'].items():
        if not re.fullmatch('[A-Za-z0-9_.-]+',cid) or set(candidate)!={'ce','origin','route'}:
            raise ValueError('Unexpected public candidate field')
        if (candidate['origin'] not in ('independent','H_baseline','A_ancestor','B_ancestor')
                or set(candidate['route'])!={'kind','source_view','wire'}
                or candidate['route']['kind']!='model' or candidate['route']['source_view'] not in ('A','B','AB')
                or candidate['route']['wire'] not in ('H','release')):raise ValueError('Unknown public candidate route')
        scores[cid]=_scores(candidate['ce'])
    groups=row['comparison_predictors']
    if set(groups)!={'selected_deployment','independent_catchup','standard_without360','mlp120','continued_mlp360','best_tree','logistic'}:
        raise ValueError('Unexpected diagnostic group')
    if set(row['comparison_ce'])!=set(groups):raise ValueError('Diagnostic score/group mismatch')
    for group,cid in groups.items():
        if cid not in scores or _scores(row['comparison_ce'][group])!=scores[cid]:raise ValueError('Derived diagnostic CE changed')
    for key,baseline,candidate in (('continued360_gain_over120','mlp120','continued_mlp360'),
        ('catchup_gain_over_standard','standard_without360','independent_catchup'),
        ('best_tree_gain_over_logistic','logistic','best_tree'),('nonlinear_gain_over_logistic','logistic','independent_catchup')):
        if row[key]!=_gain(scores,groups[baseline],groups[candidate]):raise ValueError('Derived diagnostic gain changed')
    for key in ('fit_rows','validation_rows'):
        if isinstance(row[key],bool) or not isinstance(row[key],int) or row[key]<=0:raise ValueError('Invalid row count')
    provenance=row['provenance']
    if set(provenance)!={'receipt_sha256','selection_sha256','mlp_metadata_sha256'} or set(provenance['mlp_metadata_sha256'])!={'120','360'}:
        raise ValueError('Unexpected public calibration provenance')
    for v in (provenance['receipt_sha256'],provenance['selection_sha256'],*provenance['mlp_metadata_sha256'].values()):_hash(v)
    trajectory=row['trajectory']
    if set(trajectory)!={'same_initialization_and_schedule','seed','schedule_seed','initial_state_hash','selected_epoch120','selected_epoch360','continuation','interpretation'}:
        raise ValueError('Unexpected public trajectory field')
    if trajectory['same_initialization_and_schedule'] is not True or trajectory['continuation']!=CONTINUATION:
        raise ValueError('Unverified continuation')
    for k in ('seed','schedule_seed','selected_epoch120','selected_epoch360'):
        if isinstance(trajectory[k],bool) or not isinstance(trajectory[k],int) or trajectory[k]<0:raise ValueError('Invalid trajectory count')
    _hash(trajectory['initial_state_hash'])


def build_calibration(payload):
    """Descriptive positive-control flags; no refitting or deployment selection."""
    if payload.get('phase')!='validation' or payload.get('evaluation_opened') is not False:
        raise ValueError('Calibration is a validation-only diagnostic')
    if set(payload)-{'phase','evaluation_opened','config_hash','source_hashes','resource_schedule_sha256','records','incomplete_units'}:
        raise ValueError('Unexpected public calibration payload field')
    for k in ('config_hash','resource_schedule_sha256'):_hash(payload[k])
    if 'source_hashes' in payload:
        if set(payload['source_hashes'])!=set(run.source_fingerprint('audit')):raise ValueError('Unexpected source schema')
        for value in payload['source_hashes'].values():_hash(value)
    records=payload['records'];lookup={}
    for row in records:
        _unit(row['configuration'],row['anchor'],row['role']);key=(row['configuration'],row['anchor'],row['role'])
        _public_record(row)
        if key in lookup:raise ValueError('Duplicate accepted calibration role')
        lookup[key]=row
    missing=[{'configuration':n,'anchor':a,'role':r} for n in CONTROLS for a in (0,1,2) for r in ROLES if (n,a,r) not in lookup]
    means={}
    for name in CONTROLS:
        if any((name,a,r) not in lookup for a in (0,1,2) for r in ROLES):continue
        means[name]={}
        for role in ROLES:
            rows=[lookup[(name,a,role)] for a in (0,1,2)]
            metrics={field:{w:math.fsum(r[field][w] for r in rows)/3 for w in WEIGHTS}
                     for field in ('continued360_gain_over120','catchup_gain_over_standard','best_tree_gain_over_logistic','nonlinear_gain_over_logistic')}
            metrics['comparison_ce']={group:{w:math.fsum(r['comparison_ce'][group][w] for r in rows)/3 for w in WEIGHTS}
                                      for group in rows[0]['comparison_ce']}
            metrics['selected_ids_by_anchor']={str(a):rows[a]['selection'] for a in (0,1,2)}
            means[name][role]=metrics
    informative=[];helped=[];nonlinear_helped=[]
    for name in CONTROLS[1:]:
        if name not in means or 'H' not in means:continue
        for role in ROLES:
            row=means[name][role]
            row['selected_recovery_over_H']={w:means['H'][role]['comparison_ce']['selected_deployment'][w]-row['comparison_ce']['selected_deployment'][w] for w in WEIGHTS}
            row['informative_control_balanced']=row['selected_recovery_over_H']['balanced']>0
            row['stress_helped_balanced']=row['catchup_gain_over_standard']['balanced']>0
            row['stress_helped_both_weightings']=all(row['catchup_gain_over_standard'][w]>0 for w in ('unweighted','weighted'))
            row['nonlinear_helped_balanced']=row['nonlinear_gain_over_logistic']['balanced']>0
            if row['informative_control_balanced']:
                informative.append({'configuration':name,'role':role})
                if row['stress_helped_balanced']:helped.append({'configuration':name,'role':role})
                if row['nonlinear_helped_balanced']:nonlinear_helped.append({'configuration':name,'role':role})
    diagnostic=('incomplete_calibration' if missing else 'positive_controls_not_informative' if not informative
                else 'stress_gain_demonstrated_on_informative_controls' if helped
                else 'no_stress_gain_demonstrated_on_informative_controls')
    return {'schema':1,'phase':'validation','evaluation_opened':False,'status':'incomplete' if missing else 'complete',
            'controls':list(CONTROLS),'roles':list(ROLES),'nominal_role_units':48,'accepted_role_units':len(records),
            'incomplete_role_units':missing,'records':copy.deepcopy(records),'three_anchor_means':means,
            'diagnostic_status':diagnostic,'informative_control_roles':informative,'stress_improved_informative_roles':helped,
            'nonlinear_upgrade_status':'incomplete_calibration' if missing else 'positive_controls_not_informative' if not informative
                else 'nonlinear_gain_demonstrated_on_informative_controls' if nonlinear_helped else 'no_nonlinear_gain_demonstrated_on_informative_controls',
            'nonlinear_improved_informative_roles':nonlinear_helped,
            'stress_definition':'added continued360 candidate versus the standard slate that already contains logistic, trees and fresh-initialized MLP120; nonlinear versus logistic gains reported separately',
            'thresholds':'strict positive point gain (>0), no tuned threshold or uncertainty claim',
            'scope':'predetermined validation diagnostic; failure to demonstrate stress gain is not proof of privacy or attack optimality; all candidate selections remain frozen',
            'candidate_semantics':'standard excludes only continued mlp360; tree/logit diagnostics use fixed independent candidates; deployment winner also includes legal H/A/B ancestors',
            'provenance':{k:copy.deepcopy(payload[k]) for k in ('config_hash','source_hashes','resource_schedule_sha256') if k in payload}}


def write_calibration(result,*,out_dir):
    root=Path(out_dir);names=('ATTACK_CALIBRATION.json','ATTACK_CALIBRATION.md')
    if any((root/n).exists() for n in names):raise FileExistsError('Calibration outputs are immutable')
    lines=['# Attack calibration','',result['scope'],'','Diagnostic: `'+result['diagnostic_status']+'`.','',
           'Nonlinear upgrade diagnostic: `'+result['nonlinear_upgrade_status']+'`.','',result['stress_definition'],'',
           'Positive gains below mean smaller validation CE. These are descriptive point comparisons, not a hypothesis test.',
           '', '| Control | Role | Weighting | Selected recovery over H | MLP120 minus continued360 | Standard minus catchup | Logistic minus best tree |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for name,roles in result['three_anchor_means'].items():
        for role,row in roles.items():
            for w in WEIGHTS:
                gain=row.get('selected_recovery_over_H',{}).get(w)
                values=['unavailable' if gain is None else f'{gain:.9g}',*(f'{row[k][w]:.9g}' for k in ('continued360_gain_over120','catchup_gain_over_standard','best_tree_gain_over_logistic'))]
                lines.append('| '+' | '.join([name,role,'PWGTP' if w=='weighted' else w,*values])+' |')
    root.mkdir(parents=True,exist_ok=True)
    with (root/names[0]).open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (root/names[1]).open('x') as f:f.write('\n'.join(lines)+'\n')
    return {name:run.sha(root/name) for name in names}


def generate(*,out_root=config.OUT,out_dir=None):
    return write_calibration(build_calibration(collect_calibration(out_root=out_root)),out_dir=out_root if out_dir is None else out_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out-root',type=Path,default=config.OUT);parser.add_argument('--out-dir',type=Path)
    args=parser.parse_args();print(json.dumps(generate(out_root=args.out_root,out_dir=args.out_dir),indent=2))
