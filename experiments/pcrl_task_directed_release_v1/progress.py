"""Public-safe execution inventory; does not read comparative performance."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import psutil
from .config import OUT,configuration,digest,release_ledger
from .run import atomic,now,sha,_validate_provenance


def evaluation_exposure(*,out_root=None):
    """Conservative metadata-only evidence of any evaluation attempt or opening.

    Canonical evaluate_unit persists its task-owned lock before loading test
    arrays. A failed permit can leave the same lock, so lock/attempt evidence
    deliberately sets evaluation_opened=true without claiming that loading
    certainly succeeded. No selection score, log body or prediction is read.
    """
    root=Path(OUT if out_root is None else out_root);evidence=[]
    from .branches import action33_specs,action33_controls
    from .robustness import robustness_specs
    possible=release_ledger()['records']+action33_specs()+action33_controls()+robustness_specs()
    known={hashlib.sha256(f"evaluation/{r['anchor']}/{r['configuration']}".encode()).hexdigest()+'.lock'
           for r in possible}
    for path in sorted((root/'private/locks').glob('*.lock')):
        if path.name in known:
            evidence.append({'kind':'durable_evaluation_lock','marker_sha256':sha(path)})
            continue
        try:record=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if isinstance(record.get('key'),str) and record['key'].startswith('evaluation/'):
            evidence.append({'kind':'durable_evaluation_lock','marker_sha256':sha(path)})
    for anchor in configuration()['seeds']:
        for path in sorted((root/'private/run'/f'anchor_{anchor}'/'evaluation').glob('*')):
            if path.is_dir():evidence.append({'kind':'evaluation_output_directory','anchor':anchor})
    for _ in (root/'private/logs').glob('evaluate__*.log'):
        evidence.append({'kind':'evaluation_process_log_exists'})
    for _ in (root/'private/quarantine').glob('evaluation-*'):
        evidence.append({'kind':'quarantined_evaluation_attempt'})
    path=root/'private/SCHEDULER.json'
    if path.exists():
        raw=json.loads(path.read_text())
        if any(j.get('command')=='evaluate' for j in raw.get('active_jobs',[])):
            evidence.append({'kind':'scheduler_active_evaluation'})
        if any(j.get('command')=='evaluate' and j.get('status') not in ('pending','dependency_blocked')
               for j in raw.get('jobs',[])):
            evidence.append({'kind':'scheduler_recorded_evaluation_attempt'})
    path=root/'RUN_LEDGER.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior.get('evaluation_opened') or prior.get('evaluation_attempted_or_opened'):
            evidence.append({'kind':'prior_conservative_ledger'})
    opened=bool(evidence)
    return {'evaluation_attempted_or_opened':opened,'evaluation_opened':opened,'evidence':evidence,
            'interpretation':'conservative: any durable canonical evaluation-attempt evidence counts as opened, including attempts that may have failed before test loading; false means no such evidence found, not a universal read-access proof'}


def snapshot():
    cfg=configuration();maps=[];audits=[];evaluations=[]
    for anchor in cfg['seeds']:
        root=OUT/'private/run'/f'anchor_{anchor}'
        for kind,folder,filename,stage,rows in (
                ('map','maps','ACCEPTED.json','map',maps),
                ('audit','audits','COMPLETE.json','audit',audits),
                ('evaluation','evaluation','COMPLETE.json','evaluation',evaluations)):
            for marker in sorted((root/folder).glob('*/'+filename)):
                receipt=json.loads(marker.read_text());_validate_provenance(receipt,stage)
                if receipt['anchor']!=anchor or receipt['configuration']!=marker.parent.name:
                    raise ValueError('Accepted unit inventory identity mismatch')
                row={'anchor':anchor,'configuration':receipt['configuration'],
                     'receipt_sha256':sha(marker),'seconds':receipt.get('seconds')}
                if kind=='audit':row.update(new_role_fits=receipt['new_role_fits'],reused_role_audits=receipt['reused_role_audits'])
                rows.append(row)
    path=OUT/'private/SCHEDULER.json';scheduler={}
    if path.exists():
        raw=json.loads(path.read_text());pid=raw['pid']
        try:alive='experiments.pcrl_task_directed_release_v1.scheduler' in psutil.Process(pid).cmdline()
        except psutil.Error:alive=False
        scheduler={'status':raw['status'],'verified_process_alive':alive,
                   'started_utc':raw['started_utc'],'updated_utc':raw.get('updated_utc'),
                   'workers':raw['workers'],'active_jobs':raw.get('active_jobs',[]),
                   'remaining_jobs':raw.get('remaining_jobs'),
                   'incidents_or_incomplete':[{'id':j['id'],'status':j['status'],'returncode':j.get('returncode')}
                       for j in raw['jobs'] if j['status']!='complete']}
    exposure=evaluation_exposure(out_root=OUT)
    result={'written_utc':now(),'status':'execution_inventory','config_hash':digest(cfg),
            'nominal_primary_maps':len(cfg['maps']),'nominal_primary_release_anchor_audits':len(release_ledger()['records']),
            'accepted_ACS_Q_maps':len(maps),'complete_ACS_release_audits':len(audits),
            'completed_role_audits':sum(r['new_role_fits']+r['reused_role_audits'] for r in audits),
            'new_role_fits':sum(r['new_role_fits'] for r in audits),
            'unchanged_B_role_audits_reused':sum(r['reused_role_audits'] for r in audits),
            'completed_evaluations':len(evaluations),
            'evaluation_opened':bool(evaluations) or exposure['evaluation_opened'],
            'evaluation_attempted_or_opened':bool(evaluations) or exposure['evaluation_attempted_or_opened'],
            'evaluation_exposure_evidence':exposure['evidence'],
            'evaluation_exposure_interpretation':exposure['interpretation'],
            'selection_frozen':(OUT/'SELECTION.json').exists(),
            'scheduler':scheduler,'maps':maps,'audits':audits,'evaluations':evaluations,
            'comparative_performance_read':False,
            'integrity_scope':'Receipt identity/config/source checked; complete artifact replay is a separate verification.'}
    atomic(OUT/'RUN_LEDGER.json',result)
    text=(f"# Run status\n\nUpdated {result['written_utc']}. "
          f"Accepted ACS maps: {len(maps)}; complete release/anchor audits: {len(audits)} "
          f"({result['completed_role_audits']} role audits, including {result['unchanged_B_role_audits_reused']} unchanged B-role reuses). "
          f"Completed evaluation units: {len(evaluations)}; evaluation attempted or opened (conservative): {result['evaluation_attempted_or_opened']}.\n\n"
          f"Scheduler: {scheduler.get('status','not recorded')}; live process verified: {scheduler.get('verified_process_alive',False)}. "
          "An accepted map is an execution result, not evidence of a competitive tradeoff. "
          "All current-run evaluation remains historically reused 2018 development data.\n")
    temp=OUT/'RUN_STATUS.md.tmp';temp.write_text(text);temp.replace(OUT/'RUN_STATUS.md')
    return result


if __name__=='__main__':
    r=snapshot()
    print(json.dumps({k:v for k,v in r.items() if k not in ('maps','audits','evaluations')}))
