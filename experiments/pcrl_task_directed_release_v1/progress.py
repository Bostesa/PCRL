"""Public-safe execution inventory; does not read comparative performance."""
from __future__ import annotations
import json
from pathlib import Path
import psutil
from .config import OUT,configuration,digest,release_ledger
from .run import atomic,now,sha,_validate_provenance


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
    result={'written_utc':now(),'status':'execution_inventory','config_hash':digest(cfg),
            'nominal_primary_maps':len(cfg['maps']),'nominal_primary_release_anchor_audits':len(release_ledger()['records']),
            'accepted_ACS_Q_maps':len(maps),'complete_ACS_release_audits':len(audits),
            'completed_role_audits':sum(r['new_role_fits']+r['reused_role_audits'] for r in audits),
            'new_role_fits':sum(r['new_role_fits'] for r in audits),
            'unchanged_B_role_audits_reused':sum(r['reused_role_audits'] for r in audits),
            'completed_evaluations':len(evaluations),'evaluation_opened':bool(evaluations),
            'selection_frozen':(OUT/'SELECTION.json').exists(),
            'scheduler':scheduler,'maps':maps,'audits':audits,'evaluations':evaluations,
            'comparative_performance_read':False,
            'integrity_scope':'Receipt identity/config/source checked; complete artifact replay is a separate verification.'}
    atomic(OUT/'RUN_LEDGER.json',result)
    text=(f"# Run status\n\nUpdated {result['written_utc']}. "
          f"Accepted ACS maps: {len(maps)}; complete release/anchor audits: {len(audits)} "
          f"({result['completed_role_audits']} role audits, including {result['unchanged_B_role_audits_reused']} unchanged B-role reuses). "
          f"Completed evaluation units: {len(evaluations)}.\n\n"
          f"Scheduler: {scheduler.get('status','not recorded')}; live process verified: {scheduler.get('verified_process_alive',False)}. "
          "An accepted map is an execution result, not evidence of a competitive tradeoff. "
          "All current-run evaluation remains historically reused 2018 development data.\n")
    temp=OUT/'RUN_STATUS.md.tmp';temp.write_text(text);temp.replace(OUT/'RUN_STATUS.md')
    return result


if __name__=='__main__':
    r=snapshot()
    print(json.dumps({k:v for k,v in r.items() if k not in ('maps','audits','evaluations')}))
