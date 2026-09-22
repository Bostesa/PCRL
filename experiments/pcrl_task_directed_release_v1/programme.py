"""Small orchestration entry points around immutable registered scientific fits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import psutil
from .config import OUT,configuration,release_ledger,mechanism_id,digest
from .run import atomic,now,sha


def registration_inputs():
    """Supply every registered extension, including resource-incomplete slots."""
    extensions={};controls={};anchors={}
    if (OUT/'EXTRA_CONFIGS.json').exists():
        from . import branches
        registry=branches._registration()
        for raw in registry['maps']+registry['controls']:
            spec=branches.lookup_release(raw['configuration'],raw['anchor'])
            if spec is None or spec.get('anchor')!=raw['anchor'] or spec.get('configuration')!=raw['configuration']:
                raise ValueError('Registered extension identity differs from lookup')
            anchors.setdefault(raw['configuration'],set()).add(raw['anchor'])
            target=controls if raw.get('kind')=='control' else extensions
            value={k:v for k,v in spec.items() if k not in ('id','anchor')}
            old=target.setdefault(raw['configuration'],value)
            if old!=value:raise ValueError('Registered common configuration differs across anchors')
    if (OUT/'ROBUSTNESS_CONFIGS.json').exists():
        from . import robustness
        registry=robustness._registration()
        for raw in registry['maps']:
            spec=robustness.lookup_spec(raw['configuration'],raw['anchor'])
            if spec is None or spec.get('anchor')!=raw['anchor'] or spec.get('configuration')!=raw['configuration']:
                raise ValueError('Registered robustness identity differs from lookup')
            anchors.setdefault(raw['configuration'],set()).add(raw['anchor'])
            value={k:v for k,v in spec.items() if k not in ('id','anchor')}
            old=extensions.setdefault(raw['configuration'],value)
            if old!=value:raise ValueError('Registered common robustness configuration differs across anchors')
    if any(value!={0,1,2} for value in anchors.values()):
        raise ValueError('Registered common configuration requires all three anchors')
    if set(extensions)&set(controls):raise ValueError('Configuration is registered as both map and control')
    primary={m['configuration'] for m in configuration()['maps'] if m['budget'] is not None and m['budget']>0}
    extra={n for n,s in extensions.items() if s['policy'] in ('L','C') and s['budget']>0}
    return {'candidate_ids':sorted(primary|extra),'registered_extensions':extensions,'registered_controls':controls}


def active_scientific_writers():
    found=[]
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid():continue
        try:args=process.info['cmdline'] or []
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        if ('experiments.pcrl_task_directed_release_v1.scheduler' in args
                or ('experiments.pcrl_task_directed_release_v1.run' in args
                    and any(command in args for command in ('prepare','benchmark','audit','map','evaluate')))):
            found.append(process.pid)
    return found


def completion_inventory():
    from . import run
    registered=[{'configuration':r['configuration'],'anchor':r['anchor'],'scope':'primary'}
                for r in release_ledger()['records']]
    seen={(r['configuration'],r['anchor']) for r in registered}
    if len(seen)!=len(registered):raise ValueError('Duplicate primary release/anchor unit')
    for name,scope in (('EXTRA_RESOURCE_SCHEDULE.json','branch_A'),('ROBUSTNESS_RESOURCE_SCHEDULE.json','branch_C')):
        path=OUT/name
        if not path.exists():continue
        if scope=='branch_A':
            from . import branches as branch
            lookup=branch.lookup_release
        else:
            from . import robustness as branch
            lookup=branch.lookup_spec
        registry=branch._registration();specs={r['id']:r for r in registry['maps']+registry.get('controls',[])}
        ids=json.loads(path.read_text())['unit_ids']
        if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids):
            raise ValueError('Duplicate or empty extension schedule')
        for ident in ids:
            if ident not in specs:raise ValueError('Extension schedule unit is not registered')
            raw=specs[ident];spec=lookup(raw['configuration'],raw['anchor'])
            schedule_sha=branch.require_scheduled(spec)
            key=(raw['configuration'],raw['anchor'])
            if key in seen:raise ValueError('Duplicate scheduled release/anchor unit')
            seen.add(key)
            registered.append({'configuration':key[0],'anchor':key[1],'scope':scope,
                               'resource_schedule_sha256':schedule_sha})
    for row in registered:
        marker=OUT/'private/run'/f"anchor_{row['anchor']}"/'audits'/row['configuration']/'COMPLETE.json'
        row['complete']=False;row['receipt_sha256']=None
        if marker.exists():
            # This also verifies preparation, actual channel, H ancestry and
            # any branch registration/schedule dependencies; no object is fit.
            receipt=run._audit_receipt(row['anchor'],row['configuration'])
            if (receipt.get('role_audits')!=len(configuration()['roles']['primary'])+len(configuration()['roles']['secondary'])+len(configuration()['roles']['utility'])
                    or receipt.get('registry_sha256')!=sha(marker.parent/'registry.joblib')
                    or not {'registry.joblib','summary.json'}<=set(receipt['artifact_hashes'])):
                raise ValueError('Accepted audit is missing its complete registered schema or registry')
            row['complete']=True;row['receipt_sha256']=sha(marker)
    return registered


def freeze(*,closeout_reason=None):
    """Freeze accepted validation choices/contrasts before any evaluation load."""
    from . import run
    with run.unit_lock('programme/selection_freeze'):
        return _freeze(closeout_reason=closeout_reason)


def _freeze(*,closeout_reason=None):
    from . import reporting,selection,progress
    if closeout_reason is not None:
        if not isinstance(closeout_reason,str) or not closeout_reason.strip():
            raise ValueError('A closeout reason must be a nonempty explanation')
        closeout_reason=closeout_reason.strip()
    if active_scientific_writers():raise RuntimeError('Close scientific writers before final selection freeze')
    if (OUT/'SELECTION.json').exists():raise FileExistsError('Scientific selection already frozen')
    if progress.evaluation_exposure(out_root=OUT)['evaluation_attempted_or_opened']:
        raise RuntimeError('Prior evaluation attempt/opening evidence prevents a sealed validation freeze')
    schedule=json.loads((OUT/'RESOURCE_SCHEDULE.json').read_text())
    if schedule.get('frozen_before_comparative_outcomes') is not True or schedule.get('config_hash')!=digest(configuration()):
        raise ValueError('Frozen resource schedule/configuration required')
    schedule_sha=sha(OUT/'RESOURCE_SCHEDULE.json')
    inventory=completion_inventory();missing=[r for r in inventory if not r['complete']]
    if missing and not closeout_reason:
        raise RuntimeError('Scheduled units remain incomplete; preserve completion or record a genuine closeout reason')
    validation=reporting.collect_validation()
    arguments=registration_inputs()
    selected=selection.select_validation(validation,**arguments)
    selected=reporting.bind_frozen_provenance(selected)
    if active_scientific_writers():raise RuntimeError('Scientific writers appeared during selection preparation')
    if progress.evaluation_exposure(out_root=OUT)['evaluation_attempted_or_opened']:
        raise RuntimeError('An evaluation attempt appeared during selection preparation')
    if sha(OUT/'RESOURCE_SCHEDULE.json')!=schedule_sha:
        raise ValueError('Resource schedule changed during selection preparation')
    for row in inventory:
        if row['complete']:
            marker=OUT/'private/run'/f"anchor_{row['anchor']}"/'audits'/row['configuration']/'COMPLETE.json'
            if not marker.exists() or sha(marker)!=row['receipt_sha256']:
                raise ValueError('Accepted completion receipt changed during selection preparation')
    selected['completion_at_freeze']={'records':inventory,'missing':missing,'closeout_reason':closeout_reason,
                                       'scope':'Incomplete units are not scientific failures or bad-outcome gates.'}
    result=selection.freeze_selection(selected,out_dir=OUT)
    atomic(OUT/'SELECTION_INPUTS.json',{'created_utc':now(),'arguments':arguments,
           'validation_grid_sha256':sha(OUT/'VALIDATION_GRID.json'),
           'selection_sha256':sha(OUT/'SELECTION.json'),'contrasts_sha256':sha(OUT/'CONTRASTS.json')})
    return result


def branch_a_schedule(*,resource_decision,include_outer_budgets=False):
    from . import branches
    registry=branches.register_action33()
    if not registry['registered']:
        atomic(OUT/'BRANCH_A_TRIGGER.json',registry)
        return registry
    maps=registry['maps'];controls=registry['controls'];order=[]
    # Whole middle-budget comparison and every matched family precede expansion.
    order.extend(s['id'] for s in maps if s['policy']=='U')
    order.extend(s['id'] for s in maps if s['budget']==.002)
    order.extend(s['id'] for s in controls)
    if include_outer_budgets:
        for budget in (.0005,.01):order.extend(s['id'] for s in maps if s['budget']==budget)
    return branches.freeze_action33_schedule(order,resource_decision=resource_decision)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('freeze');s.add_argument('--closeout-reason')
    s=sub.add_parser('branch-A');s.add_argument('--resource-decision',type=Path,required=True)
    s.add_argument('--include-outer-budgets',action='store_true')
    a=p.parse_args()
    if a.command=='freeze':freeze(closeout_reason=a.closeout_reason)
    else:branch_a_schedule(resource_decision=json.loads(a.resource_decision.read_text()),include_outer_budgets=a.include_outer_budgets)
