"""Metadata-only fixed historical score-law utility job preparation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_evaluation_units import BASE,PRIVATE,PUBLIC,COMMON,LEGACY,pin,read,write,make_unit,source


def prepare(input_path,output,python,amendment_path):
    index=read(pin(input_path));readers=read(source(index['reader_index']));rp=source(index['reporting_lock']);lock=read(rp)
    if lock.get('status')!='ALL_READERS_FROZEN':raise PermissionError('global reporting freeze absent')
    nomination=source(index['nomination_lock']);utility_dir=PRIVATE/'utility/R5_ORIGINAL_UTILITY_ALL40_V1';receipt=read(pin(utility_dir/'UTILITY_COMPLETE.json'))
    utility={'path':str(utility_dir/'UTILITY_INPUTS.npz'),'sha256':receipt['utility_input_sha256']};instances=[];inputs=[pin(input_path),rp,nomination,utility]
    for entry in readers['legacy_units']:
        cfg=read(source(entry['config']));law=pin(Path(entry['private_output'])/'INNER_LAW.npz')
        instances.append({'arm_id':entry['arm_id'],'seed':entry['seed'],'bank':source(cfg['bank']),'inner_law':law});inputs.extend([source(cfg['bank']),law])
    if len(instances)!=4:raise PermissionError('four frozen historical laws required')
    uid='R5_LEGACY_ORIGINAL_UTILITY_V1';cfg={'unit_id':uid,'nomination_lock':nomination,'utility_inputs':utility,'instances':instances,'public_output':str(PUBLIC/'LEGACY_ORIGINAL_UTILITY_V1.json'),'private_output':str(PRIVATE/'legacy_utility/LEGACY_ORIGINAL_UTILITY_V1.json')}
    ap=pin(amendment_path)
    unit=make_unit(cfg,'task_conditioned_representation.legacy_utility_worker',inputs=inputs,code=(*COMMON,*LEGACY,'task_conditioned_representation/utility_worker.py','task_conditioned_representation/legacy_utility_worker.py'),python=python,amendment_pin=ap,wall=180)
    return write(output,{'schema':'TCR1-prepared-historical-utility-v1','unit':unit,'instances':4,'INNER_SEX_input_absent':True,'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--python',required=True);p.add_argument('--attacks-amendment',required=True);a=p.parse_args();print(json.dumps(prepare(a.input,a.output,a.python,a.attacks_amendment)))
if __name__=='__main__':main()
