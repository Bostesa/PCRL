"""R6 queued independent engineering admission; INNER labels are never read."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '0aeab04be697c5b52d533874013e0ea40af0e9594b11a2988dd89562a4eb8cb1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,body):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(body,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--layout-only',action='store_true')
    a = ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard = assert_active_from_env()
    if guard['stage'] not in ('admission','verification'):
        raise RuntimeError('R6 independent admission requires admission/verification stage')
    import numpy as np
    import torch
    import joblib
    from .science import encoder, fold_pipeline, softmax, metrics, coverage,fit_fold,gram_bank,core_power,fixtures
    torch.set_num_threads(1)
    manifest_path = Path('.private/task_conditioned_representation/r1/SOURCE_PATHS.json')
    if sha(manifest_path)!=MANIFEST_SHA:
        raise ValueError('pinned safe source manifest changed')
    source = json.loads(manifest_path.read_text())
    pins_checked = []
    def checked(pin):
        assert_active_from_env()
        if sha(pin['path'])!=pin['sha256']:
            raise ValueError('source hash mismatch')
        pins_checked.append(pin['sha256'])
        return Path(pin['path'])
    # Closed label paths are not opened and not even hashed in this admission.
    allowed = {'OSF_DEFENSE_FIT':{'features','income','occupation','sex','references_s0','references_s1'},
               'ATTACK_FIT_NEW':{'features','references_s0','references_s1'},
               'INNER_SELECTION':{'features','references_s0','references_s1'}}
    roledata = {}
    layout = {}
    for role,names in allowed.items():
        roledata[role] = {}
        layout[role] = {}
        for name in sorted(names):
            file = checked(source['roles'][role][name])
            with np.load(file,allow_pickle=False) as data:
                roledata[role][name] = {k:data[k] for k in data.files}
                layout[role][name] = {k:{'shape':list(data[k].shape),'dtype':str(data[k].dtype)} for k in data.files}
    models = {}
    schema_names = None
    model_layout = {}
    for seed in (0,1):
        entry = source['seeds'][str(seed)]
        paths = {k:checked(v) for k,v in entry['pins'].items()}
        for pin in entry['ucal'].values(): checked(pin)
        bank_path = checked(entry['public_bank'])
        names = json.loads(bank_path.read_text())['feature_names']
        if len(names)!=83 or len(set(names))!=83 or any(n.lower().split('=')[0] in {'sex','race','income','occupation','fnlwgt','row_id','unit'} for n in names):
            raise ValueError('source public schema contains forbidden inputs')
        if schema_names is not None and names!=schema_names:raise ValueError('paired source schemas differ')
        schema_names=names
        state = torch.load(paths['model'],map_location='cpu',weights_only=True)
        heads = [joblib.load(paths[f'head{r}']) for r in (0,1)]
        models[seed] = state,heads
        model_layout[str(seed)] = {k:list(v.shape) for k,v in state.items()}
    if a.layout_only:
        atomic(a.out,{'schema':'TCR1-R6-layout-v1','status':'PASS','guard':guard,'roles':layout,'models':model_layout})
        return
    # Exact shared formats are explicit; fail rather than infer a mismatched schema.
    expected = {'OSF_DEFENSE_FIT':(15434,15428),'ATTACK_FIT_NEW':(4064,4061),'INNER_SELECTION':(2235,2234)}
    row_sets, group_sets, reports = [],[],[]
    for role,(n,g) in expected.items():
        arrays = roledata[role]['features']
        x,ids,groups = arrays['X'],arrays['row_id'],arrays['unit']
        if set(arrays)!={'X','row_id','unit','feature_names'} or list(map(str,arrays['feature_names']))!=schema_names:
            raise ValueError('role feature schema differs from pinned source public schema')
        if x.shape!=(n,83) or x.dtype!=np.float32 or not np.isfinite(x).all() or len(np.unique(groups))!=g or len(np.unique(ids))!=n:
            raise ValueError('source role count/schema mismatch')
        for name,data in roledata[role].items():
            if not np.array_equal(data['row_id'],ids) or not np.array_equal(data['unit'],groups):
                raise ValueError('source person order differs across role-safe files')
            expected_fields = {'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'} if name.startswith('references_') else {'row_id','unit',name} if name!='features' else {'X','row_id','unit','feature_names'}
            if set(data)!=expected_fields:raise ValueError('source role-safe file fields differ')
        row_sets.append(set(ids.tolist())); group_sets.append(set(groups.tolist()))
        for seed in (0,1):
            state,heads = models[seed]
            ref = roledata[role][f'references_s{seed}']
            cal = json.loads(Path(source['seeds'][str(seed)]['ucal']['record']['path']).read_text())
            task_reports = []
            for r in (0,1):
                z = encoder(state,r,x)
                w,b,classes = fold_pipeline(heads[r])
                if not np.array_equal(classes,np.arange((2,6)[r])):
                    raise ValueError('source class order mismatch')
                c = z@w.T+b
                p = softmax(c)
                deployed = heads[r].predict_proba(z)
                # Stored export keys are established by layout first.
                stored = ref[f'U0_{r+1}']
                alpha = cal['tables'][f'H-GLOBAL-TEMP|r{r+1}']['alpha']
                clipped = np.maximum(stored,1e-12)
                clipped/=clipped.sum(1,keepdims=True)
                ucal = softmax(alpha*np.log(clipped))
                err = float(np.max(np.abs(p-deployed)))
                serr = float(np.max(np.abs(deployed-stored)))
                cerr = float(np.max(np.abs(ucal-ref[f'Ucal_{r+1}'])))
                if max(err,serr,cerr)>2e-12 or not np.array_equal(p.argmax(1),deployed.argmax(1)) or not np.array_equal(p.argmax(1),ref[f'd_{r+1}']):
                    raise ValueError('independent source fold/Ucal parity failure')
                report = {'recipient':r+1,'folded_vs_source_max_abs':err,'source_vs_export_max_abs':serr,
                    'ucal_vs_export_max_abs':cerr,'alpha':alpha,'classes':classes.tolist(),'head_norm':float(np.linalg.norm(w))}
                if role=='OSF_DEFENSE_FIT':
                    labelname = ('income','occupation')[r]
                    y = roledata[role][labelname][labelname]
                    report['metrics'] = metrics(y,p)
                task_reports.append(report)
            reports.append({'role':role,'seed':seed,'rows':n,'groups':g,'tasks':task_reports})
    if any(row_sets[i]&row_sets[j] or group_sets[i]&group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('admitted role group overlap')
    fit = roledata['OSF_DEFENSE_FIT']
    groups = fit['features']['unit']; folds=fit_fold(groups)
    cov = {}
    for name,mask in [('allFIT',np.ones(len(groups),dtype=bool)),('training01',folds!=2),('selection2',folds==2)]:
        cov[name] = coverage(fit['income']['income'],fit['occupation']['occupation'],fit['sex']['sex'],groups,mask)
    laws = {}
    for g,yi,yo,s in zip(groups,fit['income']['income'],fit['occupation']['occupation'],fit['sex']['sex']):
        law=(int(yi),int(yo),int(s))
        if g in laws and laws[g]!=law:raise ValueError('duplicate-group labels contradict')
        laws[g]=law
    # Official authentic pinned core is checked against the independent NumPy derivation.
    official = Path(__file__).parent/'vendor'/'mmd_official.py'
    code = official.read_text().replace('from utils.utils import zero_diag','')
    namespace = {'zero_diag':lambda x:x-torch.diag_embed(torch.diagonal(x,dim1=-2,dim2=-1))}
    exec(compile(code,str(official),'exec'),namespace)
    rng = np.random.default_rng(42062026)
    x,y = rng.normal(size=(32,5)),rng.normal(loc=.3,size=(32,5))
    k = gram_bank(x,y,np.vstack((x,y)))
    power,mmd,var = core_power(*k)
    tk = [torch.from_numpy(v) for v in k]
    ommd,ovar,_ = namespace['h1_mean_var_gram'](*tk,m=8,is_var_computed=True,unbiased_variance=False)
    opower,_ = namespace['block_approx_test_power'](ommd,torch.sqrt(torch.maximum(ovar,torch.tensor(1e-6,dtype=torch.float64))),4,
                    torch.tensor(1.6448536269514722,dtype=torch.float64),torch.distributions.Normal(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)))
    errs = {'mmd':float(np.max(abs(mmd-ommd.numpy()))),'variance':float(np.max(abs(var-ovar.numpy()))),'power':float(np.max(abs(power-opower.numpy())))}
    if max(errs.values())>1e-12:
        raise ValueError('independent official core parity failure')
    scale_errors = []
    for factor in (1e-3,1e-2,1,1e2,1e3):
        factors = np.geomspace(factor,1/factor,5)
        sp,_,_ = core_power(*gram_bank(x*factors,y*factors,np.vstack((x,y))*factors))
        scale_errors.append(float(np.max(abs(sp-power))))
    if max(scale_errors)>1e-10:
        raise ValueError('scale invariance failure')
    atomic(a.out,{'schema':'TCR1-R6-independent-prefit-v1','status':'PASS','guard':guard,
        'source_manifest_sha256':MANIFEST_SHA,'checked_pins':len(pins_checked),'source_forward':reports,
        'coverage':cov,'fold_counts':[int((folds==i).sum()) for i in range(3)],
        'official_full_sha':'280a178abc86802cef58f99ab3eff4d2365ddacb','official_snapshot_sha256':sha(official),
        'official_core_errors':errs,'scale_fixture_max_errors':scale_errors,'independent_fixtures':fixtures(),
        'closed_roles_read':False,'INNER_task_labels_read':False,'INNER_SEX_read':False})

if __name__=='__main__': main()
