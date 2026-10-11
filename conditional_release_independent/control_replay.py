"""Independent archived control solve arithmetic; global scope explicit."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .reconstruction import digest
from .primary_replay import load_archive,matrix,certify


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued control config')
    cfg=json.loads(Path(config).read_text());results=[]
    for unit in cfg['controls']:
        entry=unit['receipt']
        if pins.get(entry['path'])!=entry['sha256'] or digest(entry['path'])!=entry['sha256']:raise PermissionError('control receipt not queued')
        receipt=json.loads(Path(entry['path']).read_text());solves=[];relaxation_bound=None
        if receipt['solver_calls']!=len(receipt['solves']) or receipt['solver_calls']+receipt['skipped_solver_slots']!=receipt['maximum_planned_solver_calls']:raise ValueError('control solve accounting mismatch')
        for item in receipt['solves']:
            solve=item['solve'];entry={'path':str(Path(unit['private_root'])/solve['archive_name']),'sha256':solve['archive_sha256']}
            if pins.get(entry['path'])!=entry['sha256']:raise PermissionError('control archive not queued')
            data=load_archive(entry);record={'archive_sha256':entry['sha256'],'stage':item.get('stage'),'success':solve['success']}
            if solve['success']:
                if item.get('stage')=='binary':
                    x=data['primal'];p=len(data['theta']);m=matrix(data,'inequality');eq=matrix(data,'equality');bounds=data['bounds']
                    violation=max(0.,float(np.max(m@x-data['inequality_rhs'],initial=0)),float(np.max(np.abs(eq@x-data['equality_rhs']),initial=0)),float(np.max(bounds[:,0]-x,initial=0)),float(np.max(x-bounds[:,1],initial=0)))
                    integer=float(np.max(np.abs(x[:p]-np.rint(x[:p])),initial=0));objective=float(data['objective']@x)
                    if max(violation,integer)>1e-7 or abs(objective-solve['certificate']['primal_objective'])>1e-7:raise ValueError('binary primal/integrality/objective replay')
                    rawentry=unit['raw_binary'];
                    if pins.get(rawentry['path'])!=rawentry['sha256']:raise PermissionError('raw binary incumbent not queued')
                    from .reconstruction import admitted_arrays
                    raw=admitted_arrays(rawentry,{'primal'})['primal'];error=float(np.max(np.abs(raw[:p]-np.rint(raw[:p])),initial=0))
                    if error>1e-7 or not np.array_equal(x[:p],np.rint(raw[:p])) or not np.array_equal(x[p:],raw[p:]):raise ValueError('binary snap changed noninteger coordinates or exceeded tolerance')
                    record.update(maximum_primal_violation=violation,maximum_integer_error=integer,unsnapped_integer_error=error,primal_objective=objective,independent_relaxation_gap_upper_bound=None if relaxation_bound is None else max(0.,objective-relaxation_bound),global_optimality_independently_proved=relaxation_bound is not None and objective-relaxation_bound<=1e-7)
                else:
                    certificate=certify(data,solve['certificate']);record['certificate']=certificate
                    if item.get('stage')=='relaxation':relaxation_bound=certificate['dual_objective']
            solves.append(record)
        results.append({'unit_id':receipt['unit_id'],'seed':receipt['seed'],'kind':receipt['kind'],'support':receipt['support'],'status':receipt['status'],'solver_calls':len(solves),'skipped_slots':receipt['skipped_solver_slots'],'solves':solves,'global_optimality_claimed':False})
    result={'schema':'CBR1-independent-control-solve-replay-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'new_optimizer_fits':0,'scope':'Independent archived LP primal/dual arithmetic and binary raw-incumbent/integrality/primal checks. Each LP certificate concerns its serialized coordinate/stage program. Alternating product and freeze-first sequential families receive no global optimality certificate. Executable laws and complete utility/guard constraints are reconstructed independently in the separate policy replay.'}
    with Path(cfg['output']).open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','units':len(results),'solves':sum(x['solver_calls'] for x in results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
