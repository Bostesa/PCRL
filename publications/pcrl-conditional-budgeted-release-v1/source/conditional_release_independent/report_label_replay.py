"""Independent INNER label identity after the global complete-reader freeze."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .reconstruction import admitted_arrays,digest
from .arithmetic import validate_group_pairing


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={e['path']:e['sha256'] for e in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued INNER label replay source')
    check({'path':str(Path(config).absolute()),'sha256':digest(config)})
    cfg=json.loads(Path(config).read_text());check(cfg['reporting_lock'])
    lock=json.loads(Path(cfg['reporting_lock']['path']).read_text())
    if lock['status']!='ALL_READERS_FROZEN' or lock['bank_count']!=20 or lock['K']!=552:raise PermissionError('global reader freeze absent')
    for e in lock['reader_freezes']:
        check(e)
        if json.loads(Path(e['path']).read_text())['status']!='FROZEN_COMPLETE':raise PermissionError('unfrozen reader bank')
    # Only after all complete-reader receipts have been independently checked.
    e=lock['inner_sex_pin'];check(e);expected=admitted_arrays(e,{'row_id','unit','sex'})
    validate_group_pairing(expected['unit'],expected['sex'])
    results=[];identities=set()
    for report in cfg['reports']:
        identity=(report['arm_id'],report['seed'])
        if identity in identities:raise ValueError('duplicate report identity')
        identities.add(identity);check(report['scores'])
        # Load only the three permitted identity/label arrays, not extra members.
        with np.load(report['scores']['path'],allow_pickle=False) as archive:
            if not all(np.array_equal(expected[k],archive[k]) for k in ('row_id','unit','sex')):raise ValueError('report labels differ from admitted INNER role')
        results.append({'arm_id':identity[0],'seed':identity[1],'INNER_people_groups_labels':'EXACT','score_law_sha256':report['scores']['sha256']})
    if len(results)!=20:raise ValueError('all twenty report laws required')
    report={'schema':'CBR1-independent-report-label-custody-v1','status':'PASS','unit_id':active['unit_id'],'report_laws_checked':20,'INNER_rows':len(expected['row_id']),'INNER_groups':len(np.unique(expected['unit'])),'source_INNER_label_sha256':e['sha256'],'results':results,'INNER_SEX_access_after_all_reader_freeze':True,'HEAD_or_old_assessment_access':False,'reader_refits':0,'scope':'Every reported score-law row, duplicate group and protected label exactly matches the allowed role-specific INNER export. Complete-reader freeze is independently checked before source labels materialize. No old assessment or HEAD arrays are opened.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','report_laws':20}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
