"""Metadata-only preparation of independently implemented queued verification.

No package importing numerical code, person-array deserialization, dispatch or
label exploration occurs in this builder. The root executes every queue unit.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('preserve existing unit preparation')
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def build_refits(root,index_path):
    index=json.loads(index_path.read_text());selected=[];sources=[pin(index_path)]
    for unit in index['units']:
        if unit['arm_id'] not in ('JOINT_CONDITIONAL','U_NEW'):continue
        directory=Path(unit['private_output']);receipt=pin(directory/'FROZEN_COMPLETE.json');frozen=json.loads(Path(receipt['path']).read_text())
        if frozen['status']!='FROZEN_COMPLETE' or frozen['INNER_SEX_opened']:raise ValueError('source reader bank not prospectively frozen')
        selected.append({'family':unit['arm_id'],'seed':unit['seed'],'directory':str(directory),'receipt_sha256':receipt['sha256']})
        sources.append(receipt)
        for file,key in [('ATT_RELEASE.npz','att_release_sha256'),('INNER_RELEASE.npz','inner_release_sha256')]:sources.append({'path':str(directory/file),'sha256':frozen[key]})
    if len(selected)!=4:raise ValueError('four independent reader refits required')
    unit_id='R6_INDEPENDENT_PAIR_REFITS_V1';public=root/'results/pcrl_task_conditioned_representations_v1/r6/INDEPENDENT_PAIR_REFITS_V1.json';private=root/'.private/task_conditioned_representation/r6/refits_v1'
    config=root/f'.private/task_conditioned_representation/r6/configs/{unit_id}.json'
    cfg={'banks':selected,'private_output':str(private),'output':str(public)};write(config,cfg);sources.append(pin(config))
    for name in ('__init__','science','readers','refit_worker'):sources.append(pin(root/f'task_conditioned_representation_independent/{name}.py'))
    deps=[u['unit_id'] for u in index['units']+index['legacy_units']]
    spec={'unit_id':unit_id,'stage':'verification','argv':['/Users/nathansamson/PCRL/.venv/bin/python','-m','task_conditioned_representation_independent.refit_worker','--spec',str(config)],'cwd':str(root),'sources':sources,'outputs':[str(public)],'deps':deps,'cpu_reserve_s':300,'max_wall_s':300,'max_rss_gib':2,'projected_output_bytes':10485760}
    destination=root/f'.private/task_conditioned_representation/r6/specs/{unit_id}.json';write(destination,spec)
    return {'unit_id':unit_id,'config':pin(config),'spec':pin(destination),'public_output':str(public),'private_output':str(private),'numerical_jobs_launched':0,'INNER_SEX_opened_or_hashed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reader-index',required=True);a=p.parse_args()
    root=Path(a.root).resolve();result=build_refits(root,Path(a.reader_index).resolve())
    write(root/'.private/task_conditioned_representation/r6/INDEPENDENT_REFIT_UNIT_INDEX_V1.json',result)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
