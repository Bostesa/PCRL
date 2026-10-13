"""Metadata-only utility/oracle and final-reader queue specification preparation.

R00 creates every lock and makes every nomination/gate/fallback decision.
This helper neither deserializes arrays/checkpoints nor executes queue jobs.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PRIVATE=BASE/'.private/task_conditioned_representation/r5'
PUBLIC=BASE/'results/pcrl_task_conditioned_representations_v1/r5'
COMMON=('task_conditioned_representation/__init__.py','task_conditioned_representation/source.py','task_conditioned_representation/scheduler.py','task_conditioned_representation/attacks.py','task_conditioned_representation/inference.py','task_conditioned_representation/attack_worker.py')
MODEL=('task_conditioned_representation/models.py','task_conditioned_representation/useful_service.py')
LEGACY=('task_conditioned_representation/legacy_attack_worker.py','conditional_release/attacks.py','conditional_release/metrics.py')


def pin(path):
    p=Path(path)
    if not p.is_absolute():p=BASE/p
    if p.is_symlink():raise PermissionError('symlink refused')
    p=p.resolve(strict=True)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':sha}


def source(item):
    """Normalize existing admitted metadata; never hash a role archive here."""
    if not isinstance(item,dict) or not isinstance(item.get('sha256'),str) or len(item['sha256'])!=64:raise ValueError('source pin malformed')
    p=Path(item['path']);p=p if p.is_absolute() else BASE/p
    low=p.as_posix().lower()
    if any(s in low for s in ('head_validation','old_assessment','osf_development_assessment','all_role')):raise PermissionError('barred source metadata')
    if 'inner' in low and p.name.lower().startswith('sex'):raise PermissionError('reporting SEX absent from this metadata phase')
    return {'path':str(p.resolve()),'sha256':item['sha256']}


def read(item):
    actual=pin(item['path'])
    if actual['sha256']!=item['sha256']:raise PermissionError('metadata JSON bytes differ')
    return json.loads(Path(actual['path']).read_text())


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return pin(p)


def unique(items):
    out={}
    for item in items:
        clean=source(item)
        if clean['path'] in out and out[clean['path']]!=clean['sha256']:raise PermissionError('conflicting source metadata')
        out[clean['path']]=clean['sha256']
    return [{'path':p,'sha256':h} for p,h in sorted(out.items())]


def amendment(index,override=None):
    if override:
        item=source(override);read(item);return item
    if index.get('attacks_amendment'):
        item=source(index['attacks_amendment']);read(item);return item
    state=json.loads((BASE/'.private/task_conditioned_representation/compute/state.json').read_text())
    candidates=[i for i in state.get('amendments',[]) if 'attacks' in i['stages']]
    if not candidates or 'attacks' not in Path(candidates[-1]['path']).name.lower():raise PermissionError('registered attacks-named amendment required')
    item=source(candidates[-1]);read(item);return item


def make_unit(config,module,*,inputs,code,python,amendment_pin,wall=1200,deps=(),phase=None):
    uid=config['unit_id'];config_pin=write(PRIVATE/'configs'/f'{uid}.json',config)
    argv=[python,'-m',module,'--spec',config_pin['path']]
    if phase is not None:argv.extend(['--phase',phase])
    sources=unique([config_pin,amendment_pin,*inputs,*[pin(BASE/name) for name in code],pin(__file__)])
    outputs=[config['public_output']]
    spec={'unit_id':uid,'stage':'attacks','cwd':str(BASE),'argv':argv,'max_wall_s':wall,'cpu_reserve_s':wall,'max_rss_gib':2,
          'projected_output_bytes':128*1024**2,'sources':sources,'deps':list(deps),'outputs':outputs,'amendment_sha256':amendment_pin['sha256']}
    spec_pin=write(PRIVATE/'specs'/f'{uid}.json',spec)
    return {'unit_id':uid,'config':config_pin,'spec':spec_pin,'private_output':config['private_output'],'public_output':config['public_output']}


def role_pins(manifest,fields):
    return [source(manifest['roles'][role][key]) for role,keys in fields for key in keys]


def prepare_utility(operational_index,output,python,attacks_amendment=None):
    index=read(pin(operational_index));manifest_pin=source(index['manifest']);manifest=read(manifest_pin)
    lock_pin=source(index['nomination_lock']);lock=read(lock_pin);science=source(index['science_lock']);read(science)
    if lock.get('status')!='FROZEN' or not lock.get('nominees'):raise PermissionError('all-family FROZEN nomination lock required')
    instances=index['instances'];associations={(i['arm_id'],i['seed']):i['checkpoint_sha256'] for i in lock['instances']}
    if len(instances)!=40 or len(associations)!=40 or len(lock['instances'])!=40:raise PermissionError('22 nominees plus18 preselected fallbacks required')
    actual={(i['arm_id'],i['seed']):i['checkpoint_sha256'] for i in instances}
    if len(actual)!=40 or actual!=associations:raise PermissionError('operational checkpoint instances differ from canonical lock')
    for arm in {i['arm_id'] for i in instances}:
        if sorted(i['seed'] for i in instances if i['arm_id']==arm)!=[0,1]:raise PermissionError('two source seeds required per arm ID')
    if sum(i['arm_id'].endswith('_DIAGNOSTIC_FALLBACK') for i in instances)!=18:raise PermissionError('distinct frozen fallback IDs required')
    ap=amendment(index,attacks_amendment);unit=index.get('utility_unit_id','R5_ORIGINAL_UTILITY_ALL40_V1')
    cfg={'unit_id':unit,'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],'nomination_lock':lock_pin['path'],
         'nomination_lock_sha256':lock_pin['sha256'],'instances':instances,'private_output':str(PRIVATE/'utility'/unit),'public_output':str(PUBLIC/f'{unit}.json')}
    checkpoint_pins=[{'path':i['checkpoint'],'sha256':i['checkpoint_sha256']} for i in instances]
    role_sources=role_pins(manifest,[('INNER_SELECTION',('features','income','occupation','references_s0','references_s1'))])
    extra=[source(i) for i in index.get('extra_sources',[])];base_inputs=[pin(operational_index),manifest_pin,lock_pin,science,*extra]
    utility=make_unit(cfg,'task_conditioned_representation.utility_worker',inputs=[*base_inputs,*checkpoint_pins,*role_sources],code=(*COMMON,*MODEL,'task_conditioned_representation/utility_worker.py'),python=python,amendment_pin=ap,wall=600,deps=index.get('deps',[]))
    uid=index.get('oracle_unit_id','R5_FIT_LABEL_ORACLE_V1');oracle_cfg={'unit_id':uid,'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],
        'nomination_lock':lock_pin['path'],'nomination_lock_sha256':lock_pin['sha256'],'private_output':str(PRIVATE/'oracle'/uid),'public_output':str(PUBLIC/f'{uid}.json')}
    oracle_sources=role_pins(manifest,[('OSF_DEFENSE_FIT',('features','income','occupation','sex')),('INNER_SELECTION',('features','income','occupation'))])
    oracle=make_unit(oracle_cfg,'task_conditioned_representation.oracle_worker',inputs=[*base_inputs,*oracle_sources],code=(*COMMON,'task_conditioned_representation/oracle_worker.py'),python=python,amendment_pin=ap,wall=120,deps=index.get('deps',[]))
    return write(output,{'schema':'TCR1-prepared-utility-oracle-v1','source_operational_index':pin(operational_index),'nomination_lock':lock_pin,'manifest':manifest_pin,
                        'utility':utility,'oracle':oracle,'canonical_instances':40,'nominee_instances':22,'fallback_instances':18,'INNER_task_labels_opened':False,'INNER_SEX_opened_or_hashed':False,'executed':False})


def prepare_readers(operational_index,output,python,attacks_amendment=None):
    index=read(pin(operational_index));manifest_pin=source(index['manifest']);manifest=read(manifest_pin)
    lock_pin=source(index['attack_lock']);lock=read(lock_pin);science=source(index['science_lock']);read(science)
    if lock.get('status')!='RELEASES_AND_ENDPOINTS_FROZEN':raise PermissionError('final release/endpoint lock required')
    instances=index['instances'];associations={(i['arm_id'],i['seed']):i['checkpoint_sha256'] for i in lock['instances']}
    actual={(i['arm_id'],i['seed']):i['checkpoint_sha256'] for i in instances}
    if len(instances)!=22 or len(actual)!=22 or actual!=associations:raise PermissionError('exact22 frozen actual feature banks required')
    if sorted(i['seed'] for i in instances if i['arm_id']=='H_RAW_CSTAR')!=[0,1]:raise PermissionError('two additive historical raw feature references required')
    ap=amendment(index,attacks_amendment);units=[];base=[pin(operational_index),manifest_pin,lock_pin,science,*[source(i) for i in index.get('extra_sources',[])]]
    role_sources=role_pins(manifest,[('ATTACK_FIT_NEW',('features','sex')),('INNER_SELECTION',('features',))])
    for i in instances:
        uid=i.get('unit_id',f"R5_FIT_{i['arm_id']}_S{i['seed']}_V1")
        cfg={'unit_id':uid,'arm_id':i['arm_id'],'seed':i['seed'],'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],
             'checkpoint':i['checkpoint'],'checkpoint_sha256':i['checkpoint_sha256'],'science_lock':science['path'],'science_lock_sha256':science['sha256'],
             'attack_lock':lock_pin['path'],'attack_lock_sha256':lock_pin['sha256'],'private_output':str(PRIVATE/'readers'/uid),'public_output':str(PUBLIC/f'{uid}_FREEZE.json')}
        result=make_unit(cfg,'task_conditioned_representation.attack_worker',inputs=[*base,{'path':i['checkpoint'],'sha256':i['checkpoint_sha256']},*role_sources],code=(*COMMON,*MODEL),python=python,amendment_pin=ap,deps=index.get('deps',[]),phase='fit')
        result.update({k:i.get(k) for k in ('arm_id','seed','point_eligible','diagnostic_only','is_nominee_or_exact_alias','exact_source_alias','selection_kind','nominee_checkpoint_sha256')})
        units.append(result)
    legacy_pin=source(index.get('legacy_source_index',pin(PRIVATE/'LEGACY_ADMITTED_METADATA.json')));legacy=read(legacy_pin);legacy_units=[]
    frozen_legacy={(i['arm_id'],i['seed']) for i in lock.get('legacy_instances',[])}
    if len(legacy['instances'])!=4 or frozen_legacy!={(i['arm_id'],i['seed']) for i in legacy['instances']}:raise PermissionError('exact four frozen legacy score contracts required')
    for i in legacy['instances']:
        uid=i.get('unit_id',f"R5_LEGACY_FIT_{i['arm_id']}_S{i['seed']}_V1")
        cfg={'unit_id':uid,'arm_id':i['arm_id'],'seed':i['seed'],'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],
             'science_lock':science['path'],'science_lock_sha256':science['sha256'],'attack_lock':lock_pin['path'],'attack_lock_sha256':lock_pin['sha256'],
             'private_output':str(PRIVATE/'legacy_readers'/uid),'public_output':str(PUBLIC/f'{uid}_FREEZE.json'),
             **{k:source(i[k]) for k in ('bank','att_law','inner_law','public_prior','law_origin_receipt')}}
        oldpins=[source(i[k]) for k in ('bank','att_law','inner_law','public_prior','law_origin_receipt')]+[source(p) for p in i.get('old_policy_pins',[])]
        result=make_unit(cfg,'task_conditioned_representation.legacy_attack_worker',inputs=[*base,legacy_pin,*oldpins,*role_sources],code=(*COMMON,*LEGACY),python=python,amendment_pin=ap,deps=index.get('deps',[]),phase='fit')
        result.update(arm_id=i['arm_id'],seed=i['seed'],contract='legacy_complete_score_wire');legacy_units.append(result)
    return write(output,{'schema':'TCR1-prepared-new-and-legacy-readers-v1','source_operational_index':pin(operational_index),'attack_lock':lock_pin,'manifest':manifest_pin,
                        'units':units,'legacy_units':legacy_units,'feature_arm_seed_banks':22,'legacy_score_arm_seed_banks':4,'maximum_new_reader_fits':726,'maximum_legacy_reader_fits':120,'INNER_SEX_opened_or_hashed':False,'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('utility','readers'),required=True);p.add_argument('--operational-index',required=True);p.add_argument('--output',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--attacks-amendment');p.add_argument('--attacks-amendment-sha256');a=p.parse_args()
    override=None
    if a.attacks_amendment:
        override=pin(a.attacks_amendment)
        if a.attacks_amendment_sha256 and override['sha256']!=a.attacks_amendment_sha256:raise PermissionError('explicit attacks amendment differs')
    print(json.dumps((prepare_utility if a.phase=='utility' else prepare_readers)(a.operational_index,a.output,a.python,override)))
if __name__=='__main__':main()
