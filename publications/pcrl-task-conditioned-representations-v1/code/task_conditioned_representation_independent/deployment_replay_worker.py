"""Independent state-only deployment export and standalone-output replay."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(a.spec).read_text())
    def checked(pin):
        assert_active_from_env(stage='verification');path=Path(pin['path'])
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('independent deployment pin differs')
        return path
    import numpy as np,torch
    from .science import own_release,wire
    from .checkpoint_io import read
    torch.set_num_threads(1)
    if len(cfg['packages'])!=22 or len({(e['family'],e['seed']) for e in cfg['packages']})!=22:raise ValueError('all22 live packages required')
    required={f'enc.{r}.{layer}.{kind}' for r in (0,1) for layer in (0,2,4) for kind in ('weight','bias')}|{f'head.{r}.{kind}' for r in (0,1) for kind in ('weight','bias')}
    reports=[]
    for entry in cfg['packages']:
        meta=json.loads(checked(entry['manifest']).read_text())
        if set(meta)!={'schema','feature_names','feature_schema_sha256','class_orders','model_sha256','source_checkpoint_sha256','provenance','contract','parameter_count'} or meta['schema']!='TCR1-raw83-own-head-deployment-v1':raise ValueError('deployment metadata contains non-service state')
        if meta['class_orders']!=[[0,1],list(range(6))] or meta['parameter_count']!=21288 or meta['source_checkpoint_sha256']!=entry['checkpoint']['sha256']:raise ValueError('deployment class/parameter/source association differs')
        names=meta['feature_names'];hash_names=hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if hash_names!=meta['feature_schema_sha256'] or len(names)!=83 or len(set(names))!=83:raise ValueError('deployment ordered raw schema differs')
        state=torch.load(checked({'path':str(Path(entry['manifest']['path']).parent/'model_state.pt'),'sha256':meta['model_sha256']}),map_location='cpu',weights_only=True)
        checkpoint=read(checked(entry['checkpoint']))
        if set(state)!=required or sum(v.numel() for v in state.values())!=21288:raise ValueError('deployment includes optimizer/reference or unexpected model state')
        for key,value in state.items():
            if value.dtype!=(torch.float32 if key.startswith('enc.') else torch.float64) or not torch.isfinite(value).all() or not torch.equal(value,checkpoint['model'][key]):raise ValueError('deployment saved encoder/head state differs')
        with np.load(checked(entry['input']),allow_pickle=False) as z:
            if set(z.files)!={'X','feature_names'} or list(map(str,z['feature_names']))!=names:raise ValueError('deployment input has labels or reordered schema')
            released=own_release(state,z['X']);rows=len(z['X'])
        with np.load(checked(entry['expected']),allow_pickle=False) as z:
            if set(z.files)!={'recipient1','recipient2','score1','score2'}:raise ValueError('live release exposes unexpected fields')
            errors=[]
            for r in (1,2):
                full=wire(released,r);scores=np.column_stack((released[f'p{r}'],released[f'hard{r}']))
                errors.extend([float(np.max(abs(full-z[f'recipient{r}']))),float(np.max(abs(scores-z[f'score{r}'])))])
        with np.load(checked(entry['cli_output']),allow_pickle=False) as z:
            if set(z.files)!={'recipient1','recipient2'}:raise ValueError('CLI release fields differ')
            errors.extend(float(np.max(abs(wire(released,r)-z[f'recipient{r}']))) for r in (1,2))
        if max(errors)>1e-10:raise ValueError('independent standalone deployed outputs differ')
        reports.append({'family':entry['family'],'seed':entry['seed'],'rows':rows,'maximum_output_error':max(errors),'source_state_exact':True,'only_saved_encoders_and_own_heads':True})
    out={'schema':'TCR1-independent-live-deployment-replay-v1','status':'PASS','guard':guard,'packages':reports,'labels_or_SEX_opened':False,'candidate_deployment_code_imported':False}
    p=Path(cfg['output']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
