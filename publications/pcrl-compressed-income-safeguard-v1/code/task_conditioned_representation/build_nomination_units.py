"""Metadata-only preparation of a queue-only FIT nomination draft."""
import argparse,json,sys
from pathlib import Path
from .build_training_units import BASE,PRIVATE,PUBLIC,CODE,pin,read,write


def prepare(initial_index,protection_index,output,python=sys.executable,unit_id='R3_NOMINATION_DRAFT_V1'):
    initial=read(initial_index);protected=read(protection_index)
    if initial['manifest']!=protected['manifest'] or initial['science_lock']!=protected['science_lock']:raise ValueError('training source/lock differs')
    fits=[u for u in initial['units'] if u['family'] in ('U_NEW','U_TASKPOWER')]+protected['units']
    if len(fits)!=46 or protected['unsupported']:raise ValueError('complete registered continuation comparison required')
    result_pins=[];checkpoint_pins=[]
    for entry in fits:
        result=pin(Path(entry['private_output'])/'PRIVATE_FIT_RESULT.json');body=read(result)
        if (body['family'],body['seed'])!=(entry['family'],entry['seed']):raise ValueError('FIT metadata instance mismatch')
        result_pins.append(result)
        checkpoint_pins.extend(pin(r['checkpoint'],r['checkpoint_sha256']) for r in body['checkpoints'])
    cfg={'unit_id':unit_id,'manifest':initial['manifest'],'science_lock':initial['science_lock'],'fit_results':result_pins,
         'private_output':str(PRIVATE/'FIT_NOMINATION_DRAFT_V1.json'),'public_output':str(PUBLIC/'FIT_NOMINATION_DRAFT_V1.json')}
    config=write(PRIVATE/'configs'/f'{unit_id}.json',cfg)
    sources=[config,initial_index,protection_index,initial['manifest'],initial['science_lock'],*result_pins,*checkpoint_pins]
    sources.extend(pin(BASE/name) for name in CODE)
    sources.extend(pin(BASE/'task_conditioned_representation'/name) for name in ('build_training_units.py','build_nomination_units.py','nomination_worker.py'))
    manifest=read(initial['manifest'])
    for seed in (0,1):
        source=manifest['seeds'][str(seed)];sources.extend(source['pins'].values());sources.append(source['public_bank'])
    state_path=BASE/'.private/task_conditioned_representation/compute/state.json';state=json.loads(state_path.read_text());latest=None
    for amendment in state['amendments']:
        pin(amendment['path'],amendment['sha256'])
        if 'verification' in amendment['stages']:latest=amendment;sources.append({'path':amendment['path'],'sha256':amendment['sha256']})
    unique={}
    for item in sources:
        actual=pin(item['path'],item['sha256']);unique[actual['path']]=actual
    spec={'unit_id':unit_id,'stage':'verification','cwd':str(BASE),
          'argv':[python,'-m','task_conditioned_representation.nomination_worker','--spec',config['path']],
          'max_wall_s':600,'cpu_reserve_s':600,'max_rss_gib':2,'projected_output_bytes':8*1024**2,
          'sources':list(unique.values()),'deps':[u['unit_id'] for u in fits],
          'outputs':[cfg['private_output'],cfg['public_output']]}
    if latest:spec['amendment_sha256']=latest['sha256']
    specification=write(PRIVATE/'specs'/f'{unit_id}.json',spec)
    return write(output,{'schema':'TCR1-prepared-nomination-unit-v1','unit_id':unit_id,'config':config,'spec':specification,'fit_results':46,'checkpoints':len(checkpoint_pins),'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial-index',required=True);p.add_argument('--initial-index-sha256',required=True);p.add_argument('--protection-index',required=True);p.add_argument('--protection-index-sha256',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(prepare(pin(a.initial_index,a.initial_index_sha256),pin(a.protection_index,a.protection_index_sha256),a.output,a.python)))
if __name__=='__main__':main()
