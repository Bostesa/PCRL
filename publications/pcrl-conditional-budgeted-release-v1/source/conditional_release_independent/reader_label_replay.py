"""Independent permitted ATT label identity across every frozen reader bank."""
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
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued ATT label replay source')
    check({'path':str(Path(config).absolute()),'sha256':digest(config)})
    cfg=json.loads(Path(config).read_text());check(cfg['manifest'])
    source=json.loads(Path(cfg['manifest']['path']).read_text())
    e=source['roles']['ATTACK_FIT_NEW']['sex'];check(e)
    expected=admitted_arrays(e,{'row_id','unit','sex'})
    validate_group_pairing(expected['unit'],expected['sex'])
    results=[];identities=set()
    for bank in cfg['banks']:
        identity=(bank['arm_id'],bank['seed'])
        if identity in identities:raise ValueError('duplicate reader-bank identity')
        identities.add(identity);check(bank['freeze']);check(bank['training'])
        receipt=json.loads(Path(bank['freeze']['path']).read_text())
        if receipt['status']!='FROZEN_COMPLETE' or receipt['input_sha256']!=bank['training']['sha256']:raise PermissionError('ATT training archive is not frozen')
        actual=admitted_arrays(bank['training'],{'row_id','unit','code_r1','code_r2','mass','sex','fit','tune'})
        if not all(np.array_equal(expected[k],actual[k]) for k in ('row_id','unit','sex')):raise ValueError('frozen reader labels differ from admitted ATT role')
        results.append({'arm_id':identity[0],'seed':identity[1],'allowed_ATT_people_labels':'EXACT','reader_freeze_sha256':bank['freeze']['sha256']})
    if len(results)!=20:raise ValueError('all twenty frozen reader banks required')
    report={'schema':'CBR1-independent-reader-label-custody-v1','status':'PASS','unit_id':active['unit_id'],'banks_checked':20,'ATT_rows':len(expected['row_id']),'ATT_groups':len(np.unique(expected['unit'])),'source_ATT_label_sha256':e['sha256'],'results':results,'INNER_SEX_access':False,'HEAD_or_old_assessment_access':False,'reader_refits':0,'scope':'Exact row, duplicate-group and protected-label identity between every frozen reader training archive and the permitted role-specific ATT label export. Prediction/count replay is separate.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','banks':20}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
