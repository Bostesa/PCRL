"""Independent numerical recovery from backed-up code, weights and public wires.

No original data source, training checkpoint, task label, SEX or candidate
scientific runner is used. The mature authenticated queue guard is shared.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path

FAMILIES=('U_NEW','JOINT_CONDITIONAL','LOCAL_TWIN','SEQ12','SEQ21','H_RAW_CSTAR')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification');cfg=json.loads(Path(a.spec).read_text());root=Path(cfg['restore_root']).resolve()
    if not Path(__file__).resolve().is_relative_to(root):raise PermissionError('execute independently backed-up implementation')
    def checked(pin):
        assert_active_from_env(stage='verification');path=Path(pin['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:raise PermissionError('restore-only exact pin required')
        return path
    import numpy as np,torch
    from .science import encoder
    from .readers import features,predict,load_saved_reader
    torch.set_num_threads(1)
    if any(not Path(mod.__file__).resolve().is_relative_to(root) for name,mod in sys.modules.items() if name.startswith('task_conditioned_representation_independent') and getattr(mod,'__file__',None)):raise PermissionError('independent helper loaded outside restored code')
    if len(cfg['packages'])!=12 or {(e['family'],e['seed']) for e in cfg['packages']}!={(f,s) for f in FAMILIES for s in (0,1)}:raise ValueError('predeclared twelve-package recovery scope differs')
    releases={};results=[]
    for entry in cfg['packages']:
        meta=json.loads(checked(entry['manifest']).read_text())
        if meta['schema']!='TCR1-raw83-own-head-deployment-v1' or meta['class_orders']!=[[0,1],list(range(6))] or meta['parameter_count']!=21288:raise ValueError('restored deployment definition differs')
        names=meta['feature_names'];expected_names=hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if expected_names!=meta['feature_schema_sha256'] or len(names)!=83 or len(set(names))!=83:raise ValueError('restored ordered raw83 schema differs')
        state=torch.load(checked({'path':str(Path(entry['manifest']['path']).parent/'model_state.pt'),'sha256':meta['model_sha256']}),map_location='cpu',weights_only=True)
        required={f'enc.{r}.{layer}.{kind}' for r in (0,1) for layer in (0,2,4) for kind in ('weight','bias')}|{f'head.{r}.{kind}' for r in (0,1) for kind in ('weight','bias')}
        if set(state)!=required or sum(v.numel() for v in state.values())!=21288 or any(v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('restored state is not only actual encoders and own heads')
        with np.load(checked(entry['input']),allow_pickle=False) as z:
            if set(z.files)!={'X','feature_names'} or list(map(str,z['feature_names']))!=names:raise ValueError('restore input has hidden labels or schema changes')
            raw=z['X'];full=[];output=[]
            # Declared float32 encoders and float64 affine/centering/softmax law,
            # implemented through library primitives without candidate models.
            for r in (0,1):
                z=encoder(state,r,raw)
                with torch.no_grad():
                    logits=torch.nn.functional.linear(torch.from_numpy(z),state[f'head.{r}.weight'],state[f'head.{r}.bias'])
                    logits=logits-logits.mean(1,keepdim=True);prob=torch.softmax(logits,1).numpy()
                decision=prob.argmax(1);full.append(np.column_stack((z,logits.numpy(),prob,decision)));output.append(np.column_stack((prob,decision)))
        with np.load(checked(entry['expected']),allow_pickle=False) as z:
            if set(z.files)!={'recipient1','recipient2','score1','score2'}:raise ValueError('restored release target fields differ')
            errors=[float(np.max(abs(full[r]-z[f'recipient{r+1}']))) for r in (0,1)]+[float(np.max(abs(output[r]-z[f'score{r+1}']))) for r in (0,1)]
        if max(errors)>1e-10:raise ValueError('independent backed-up release differs')
        reader_results=[]
        for reader in entry['readers']:
            model=load_saved_reader(checked(reader['model']))
            prediction=predict(model,features(full[0],full[1],reader['model_view'],reader['name']))
            if reader['orientation']==-1:prediction=1-prediction
            with np.load(checked(reader['target']),allow_pickle=False) as z:target=z[reader['score_field']]
            error=float(np.max(abs(prediction-target)))
            if error>1e-10:raise ValueError('independent backed-up reader predictions differ')
            reader_results.append({'view':reader['view'],'reader':reader['name'],'role':reader['role'],'maximum_score_error':error})
        releases[(entry['family'],entry['seed'])]=full
        results.append({'family':entry['family'],'seed':entry['seed'],'rows':len(raw),'maximum_wire_error':max(errors),'readers':reader_results})
    if len(cfg['independent_refits'])!=4:raise ValueError('all four independently refitted readers required')
    refits=[]
    for entry in cfg['independent_refits']:
        full=releases[(entry['family'],entry['seed'])];model=load_saved_reader(checked(entry['model']))
        prediction=predict(model,features(full[0],full[1],'pair',entry['name']))
        if entry['orientation']==-1:prediction=1-prediction
        with np.load(checked(entry['target']),allow_pickle=False) as z:target=z['score']
        error=float(np.max(abs(prediction-target)))
        if error>1e-10:raise ValueError('backed-up independently refitted reader differs')
        refits.append({'family':entry['family'],'seed':entry['seed'],'reader':entry['name'],'maximum_score_error':error})
    if any((n.startswith('task_conditioned_representation.') and n!='task_conditioned_representation.scheduler') or n=='conditional_release' or n.startswith('conditional_release.') for n in sys.modules):raise ImportError('scientific candidate runner imported in independent recovery')
    result={'schema':'TCR1-independent-backup-only-numerical-replay-v1','status':'PASS','guard':guard,'packages':results,'independent_refits':refits,'original_sources_used':False,'training_checkpoints_used':False,'task_or_SEX_labels_used':False,'all_numerical_inputs_and_independent_code_within_restore_root':True,'scope':'six predeclared service families both seeds; per-view LR and actual strongest frozen reader; four independent refits'}
    out=Path(cfg['output']);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
