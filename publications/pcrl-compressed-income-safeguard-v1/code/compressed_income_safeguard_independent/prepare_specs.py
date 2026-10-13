"""Metadata-only queue specs; never imports science or hashes numerical files."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.private/compressed_income_safeguard/r3'
PUBLIC=ROOT/'results/pcrl_compressed_income_safeguard_v1/r3'
ARMS=('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY')


def metadata_pin(path):
    path=Path(path).absolute()
    if path.suffix not in ('.json','.py','.md','.txt'):
        raise PermissionError('metadata builder refuses numerical/log hashing')
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise PermissionError('metadata symlink refused')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def receipt_pin(path,digest):
    """Carry an admitted digest without reading model/tensor/log/array bytes."""
    path=Path(path).absolute()
    if not isinstance(digest,str) or len(digest)!=64 or not path.is_file():raise ValueError('existing receipt-bound artifact required')
    return {'path':str(path),'sha256':digest}


def write(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w' if replace else 'x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    path.chmod(0o600)


def training(*,amendment=None,replace=False):
    uid='CIS_INDEPENDENT_TRAINING_ACCOUNTING_V1';state=json.loads((ROOT/'.private/compressed_income_safeguard/compute/state.json').read_text())
    if uid in state['units']:raise PermissionError('queued unit identity is immutable; prepare a new version')
    for seed in (0,1):
        if state['units'][f'CIS_TRAIN_S{seed}_V1']['status']!='done':raise ValueError('both actual source-seed jobs must be done')
    revisions=[a for a in state['amendments'] if 'verification' in a['stages']]
    if amendment:
        requested=metadata_pin(amendment)
        matches=[a for a in revisions if a['path']==requested['path'] and a['sha256']==requested['sha256']]
        if len(matches)!=1:raise ValueError('ROOT must register the requested verification amendment first')
        revision=matches[0]
    else:revision=revisions[-1]
    revision_pin={'path':revision['path'],'sha256':revision['sha256']}
    config={'unit_id':uid,'public_output':str(PUBLIC/'TRAINING_ACCOUNTING_V1.json'),'packages':[],'normalizations':[]}
    sources=[revision_pin,metadata_pin(ROOT/'results/pcrl_compressed_income_safeguard_v1/SCIENCE_LOCK.json')]
    for seed in (0,1):
        base=ROOT/f'.private/compressed_income_safeguard/training/S{seed}'
        top=json.loads((base/'SUMMARY.json').read_text());sources.append(metadata_pin(base/'SUMMARY.json'))
        normalization=json.loads((base/'NORMALIZATION.json').read_text())
        npin=metadata_pin(base/'NORMALIZATION.json');tpin=receipt_pin(base/'NORMALIZATION_TENSORS.pt',normalization['normalization_tensor_sha256'])
        config['normalizations'].append({'seed':seed,'receipt':npin,'tensors':tpin});sources.extend((npin,tpin))
        canonical=hashlib.sha256(json.dumps(normalization,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for arm in ARMS:
            summary=top['arms'][arm]
            if summary['normalization_receipt_sha256']!=canonical:raise ValueError('metadata normalization receipt binding differs')
            summary_pin=metadata_pin(base/arm/'SUMMARY.json');log_pin=receipt_pin(base/arm/'UPDATE_LOG.jsonl',summary['update_log_sha256'])
            config['packages'].append({'arm_id':arm,'seed':seed,'summary':summary_pin,'update_log':log_pin});sources.extend((summary_pin,log_pin))
    code=(ROOT/'compressed_income_safeguard_independent'/name for name in ('__init__.py','training_worker.py','safeguard.py','replay.py','registry.py','prepare_specs.py'))
    sources.extend(metadata_pin(p) for p in code)
    sources.extend(metadata_pin(ROOT/'compressed_income_safeguard'/name) for name in ('__init__.py','scheduler.py','admission.py','source.py'))
    config_path=PRIVATE/'configs'/f'{uid}.json';write(config_path,config,replace=replace);sources.append(metadata_pin(config_path))
    unique={}
    for item in sources:
        if item['path'] in unique and unique[item['path']]!=item['sha256']:raise ValueError('conflicting source metadata pins')
        unique[item['path']]=item['sha256']
    python=state['units']['CIS_TRAIN_S0_V1']['spec']['argv'][0]
    spec={'unit_id':uid,'stage':'verification','cwd':str(ROOT),'argv':[python,'-m','compressed_income_safeguard_independent.training_worker','--spec',str(config_path)],
          'max_wall_s':300,'cpu_reserve_s':300,'max_rss_gib':2,'projected_output_bytes':2*1024**2,
          'deps':['CIS_TRAIN_S0_V1','CIS_TRAIN_S1_V1'],'env':{},'amendment_sha256':revision['sha256'],
          'sources':[{'path':path,'sha256':digest} for path,digest in sorted(unique.items())],
          'outputs':[config['public_output']]}
    spec_path=PRIVATE/'specs'/f'{uid}.json';write(spec_path,spec,replace=replace)
    return {'schema':'CIS1-independent-training-spec-preparation-v1','status':'PREPARED_NOT_ENQUEUED',
            'unit_id':uid,'config':metadata_pin(config_path),'spec':metadata_pin(spec_path),
            'bound_verification_amendment':revision_pin,'package_count':8,'normalization_receipts':2,
            'numerical_artifact_bytes_read_or_hashed':False,'numerical_jobs_launched':0,
            'refresh_if_ROOT_changes_guard_or_latest_verification_amendment':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('training',));parser.add_argument('--amendment');parser.add_argument('--replace',action='store_true');args=parser.parse_args()
    result=training(amendment=args.amendment,replace=args.replace)
    index=PRIVATE/'TRAINING_SPEC_INDEX.json';write(index,result,replace=args.replace)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
