"""Bind required saved objects to historical hashes; no inference or fitting."""
from __future__ import annotations
import argparse, gzip, json, time, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from experiments.acs_transfer_data import sha_file
from experiments.acs_locked_evaluation import ARMS, SCOPES, TASKS, atomic_json
from scripts.verify_acs_bottleneck_scores import state_hash
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1'
OUT=ROOT/'results/redesign_20260910_acs_locked_evaluation_v1'


def read(p):
    p=Path(p)
    if p.exists():return json.loads(p.read_text())
    return json.loads(gzip.decompress(p.with_suffix(p.suffix+'.gz').read_bytes()))


def relative(p):
    p=Path(p)
    if p.is_absolute():
        try:return str(p.relative_to(ROOT))
        except ValueError:return str(p.relative_to('/Users/nathansamson/PCRL'))
    return str(p)


def run(out=OUT):
    if (out/'FROZEN_OBJECTS.json').exists():
        saved=read(out/'FROZEN_OBJECTS.json')
        for path,r in saved['files'].items():
            assert sha_file(ROOT/path)==r['sha256'],path
        for path,r in saved['candidates'].items():
            names=('prior.npz',) if r['family']=='prior' else ('preprocessing.npz','model.pt' if r['family']=='mlp' else 'model.joblib')
            for name in names:
                assert (ROOT/path/name).exists(),str(ROOT/path/name)
        print(json.dumps(saved['counts']));return saved
    tick=time.perf_counter();expected={};manifest_sources={}
    manifests=[PARENT/'LOCAL_ARTIFACTS.json',*ROOT.glob('results/redesign_*acs*/seed_*/local_artifacts.json'),
               *ROOT.glob('results/redesign_*acs*/static/seed_*/local_artifacts.json'),
               *ROOT.glob('results/redesign_*acs*/extended/seed_*/local_artifacts.json')]
    for f in manifests:
        v=read(f);rows=v.get('omitted_files',[]) if isinstance(v,dict) else v
        manifest_sources[relative(f if f.exists() else f.with_suffix('.json.gz'))]=sha_file(f if f.exists() else f.with_suffix('.json.gz'))
        for r in rows:
            if isinstance(r,dict) and 'path' in r and 'sha256' in r:
                path=relative(r['path']);h=r['sha256']
                if path in expected and expected[path]!=h:raise ValueError('Conflicting historical manifests: '+path)
                expected[path]=h
    expected.update(read(PARENT/'REUSE_MANIFEST.json')['historical_hashes'])
    for r in read(PARENT/'EXPORT_MANIFEST.json')['exports']:
        expected[relative(PARENT/r['original'])]=r['original_sha256']
    reviewed=read(out/'START.json')['reviewed_head']
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',reviewed],cwd=ROOT,text=True).splitlines())
    identities=read(PARENT/'PREFIT_IDENTITY.json');cfg=read(PARENT/'config.json')
    files={};candidates={};systems=[];references=[];controls=[];missing=[]
    def bind(path,required=True,expected_hash=None):
        p=relative(path);f=ROOT/p
        if p in files:return p
        old=expected_hash or expected.get(p)
        if old is None and p in tracked:
            old=hashlib.sha256(subprocess.check_output(['git','show',reviewed+':'+p],cwd=ROOT)).hexdigest()
        if not f.exists():
            missing.append({'path':p,'required':required,'expected_sha256':old});return p
        actual=sha_file(f)
        if old is not None and old!=actual:raise ValueError('Historical object hash mismatch: '+p)
        files[p]={'sha256':actual,'bytes':f.stat().st_size,'historical_sha256':old,
                  'historical_hash_verified':old is not None,'identity_basis':'historical manifest' if old else 'context metadata equality checked against historical selection record; no separate original file hash'}
        return p
    def candidate(path,required=True,metadata_hash=None):
        p=relative(path)
        if p in candidates:return p
        f=ROOT/p
        if not (f/'metadata.json').exists():bind(f/'metadata.json',required);return p
        meta=read(f/'metadata.json');bind(f/'metadata.json',required,metadata_hash)
        names=('prior.npz',) if meta['family']=='prior' else ('preprocessing.npz','model.pt' if meta['family']=='mlp' else 'model.joblib')
        for name in names:bind(f/name,required)
        fields=('family','n_classes','class_schema','selected_epoch','selected_state_hash','selected_optimizer_steps',
                'input_dim','fit_rows','validation_rows','fit_support','fit_coverage_complete','fit_hashes','validation_hashes',
                'seed','selection','checkpoint_criterion','audit_kind','source_state_hash','parameters','probabilities','fit_label_hash','pseudocount_per_class')
        record={k:meta[k] for k in fields if k in meta};record['required']=required
        if (f/'preprocessing.npz').exists():
            with np.load(f/'preprocessing.npz') as a:
                record['preprocessing']={'path':relative(f/'preprocessing.npz'),'sha256':sha_file(f/'preprocessing.npz'),
                    'arrays':{k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in a.files},
                    'transform':'(raw float64 projected input - saved mean) / saved scale; internal model float32 cast retained'}
        if (f/'model.pt').exists():
            cp=torch.load(f/'model.pt',map_location='cpu',weights_only=True)
            actual=state_hash(cp['state']);assert actual==meta['selected_state_hash'],p
            record['selected_tensor_hash_verified']=True
        if (f/'prior.npz').exists():
            with np.load(f/'prior.npz') as a:record['prior_probabilities']=a['probabilities'].tolist()
        candidates[p]=record;return p
    for name in ('PROTOCOL.md','NEXT_DESIGN.md','RESEARCH_DECISION.md','config.json','REPRODUCTION.md','LOCAL_ARTIFACTS.md',
                 'PREFIT_IDENTITY.json','REUSE_MANIFEST.json','ANCHOR_PARITY.md','ANCHOR_PARITY.json','INPUT_SCHEMA.json',
                 'SOURCE_FEASIBILITY.md','MATCHING_ANALYSIS.md','AUDIT_FINDINGS.md','comparison_rules.json','RELEASE_MANIFEST.json'):
        bind(PARENT/name)
    for s in range(3):
        d=PARENT/f'seed_{s}';i=identities[str(s)];transfer=ROOT/cfg['parent_results']/f'seed_{s}'
        upstream={n:bind(transfer/n) for n in ('preprocessing.json','release_maps.joblib','source_selection.json','split_rows.npz')}
        teacher=ROOT/cfg['selective_reference_results']/'static'/f'seed_{s}'/'teachers'
        for n in ('teachers.json','map_E.npz'):bind(teacher/n)
        for n in ('pca.npz','anchors.npz','teacher.npz','split_rows.npz'):bind(d/n)
        anchors={t:candidate(ROOT/r['path'],metadata_hash=sha_file(ROOT/r['path']/'metadata.json')) for t,r in i['anchors'].items()}
        selection=read(ROOT/cfg['reference_results']/f'seed_{s}'/'selection_before_test.json')
        bind(ROOT/cfg['reference_results']/f'seed_{s}'/'selection_before_test.json')
        ss=read(transfer/'source_selection.json')
        bind(transfer/'source'/f"encoder_{ss['selected_lr']}"/'selected.pt',False)
        bind(transfer/'source'/f"tree_{ss['selected_tree_leaves']}"/'source_tree_bank.joblib',False)
        for ref in ('E_pca','B_rich_bank','C_tree_bank'):
            k=f'transfer/{ref}/same_residence';cid=selection['head_selections'][k]
            path=candidate(ROOT/cfg['reference_results']/f'seed_{s}'/'fitted'/k/cid,False)
            if (ROOT/path/'metadata.json').exists():
                assert read(ROOT/path/'metadata.json')==selection['fitting_records'][k]['candidates'][cid]
            references.append({'seed':s,'reference':ref,'target':'same_residence','candidate_id':cid,'path':path,
                               'selection_provenance':relative(ROOT/cfg['reference_results']/f'seed_{s}'/'selection_before_test.json')})
        cd=ROOT/cfg['coalition_reference_results']/f'seed_{s}'/'controls';cc=read(cd/'selection_before_test.json');bind(cd/'selection_before_test.json')
        for t in ('SEX','RAC1P',*TASKS):
            path=candidate(cd/'fitted/prior'/t)
            controls.append({'seed':s,'role':'prior','target':t,'path':path,'metadata':cc['prior_metadata'][t]})
            for b in ('120','360'):
                cid=cc['selection'][b][t];path=candidate(cc['paths'][t][b][cid],False)
                controls.append({'seed':s,'role':'exposed_diagnostic','target':t,'budget':int(b),'candidate_id':cid,'path':path,
                                 'access':'true target one-hot diagnostic only; not a recipient release'})
        for c in ARMS:
            dest=d/c;selection_path=dest/'selection_before_test.json';audit_path=dest/'audits/audit_selection.json'
            u=read(selection_path);a=read(audit_path);bind(selection_path);bind(audit_path);us=[];aud=[];witness=[]
            for role,cid in u['utility'].items():
                rec=u['utility_metadata'][role]
                path=Path(rec['alias'])/cid if 'alias' in rec else dest/'fitted/utility'/role/cid
                path=candidate(path)
                assert read(ROOT/path/'metadata.json')==rec['candidates'][cid]
                us.append({'role':role,'candidate_id':cid,'path':path,'space':'wire','projection_columns':None})
            for b,roles in a['selections'].items():
                for role,scopes in roles.items():
                    for scope,cid in scopes.items():
                        rec=a['candidates'][b][role][cid];path=candidate(rec['base_candidate_directory'],metadata_hash=rec['base_metadata_sha256'])
                        aud.append({'role':role,'budget':int(b),'scope':scope,'candidate_id':cid,'path':path,
                                    **{k:rec[k] for k in ('space','projection_columns','source_view','source_candidate_id','coordinate_route') if k in rec}})
            if c!='H':
                h=read(d/'H/audits/audit_selection.json')
                for b,roles in h['selections'].items():
                    for role,scopes in roles.items():
                        for scope,cid in scopes.items():
                            inherited='anchor__'+cid;v,t=role.split('/')
                            rec=a['candidates'][b][role][inherited]
                            path=candidate(rec['base_candidate_directory'],metadata_hash=rec['base_metadata_sha256'])
                            witness.append({'role':role,'budget':int(b),'scope':scope,'candidate_id':inherited,'H_candidate_id':cid,
                                            'path':path,'space':rec['space'],'projection_columns':rec['projection_columns']})
            systems.append({'seed':s,'condition':c,'dimensions':[4,2,6] if c=='H' else [20,2,22],
                            'checkpoint':bind(d/'training'/c/'final.pt'),'checkpoint_rule':'fixed final mapper; historical validation-selected utilities/attacks',
                            'teacher':relative(teacher/'map_E.npz') if c=='E' else None,'E_dimension':i['E_dimension'] if c=='E' else None,
                            'upstream':upstream,'anchors':anchors,'utility_selection_source':relative(selection_path),'audit_selection_source':relative(audit_path),
                            'utilities':us,'audits':aud,'H_witnesses':witness})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'reviewed_commit':read(out/'START.json')['reviewed_head'],
            'root_resolution':'recorded /Users/nathansamson/PCRL prefix remapped to current repository root',
            'systems':systems,'candidates':candidates,'files':files,'historical_manifests':manifest_sources,
            'references':references,'controls':controls,'missing_objects':missing,'required_missing':sum(x['required'] for x in missing),
            'counts':{'systems':len(systems),'utility_endpoints':sum(len(x['utilities']) for x in systems),
                      'audit_selections':sum(len(x['audits']) for x in systems),'H_projection_witnesses':sum(len(x['H_witnesses']) for x in systems),
                      'unique_candidates':len(candidates),'files':len(files),'historically_hash_verified_files':sum(x['historical_hash_verified'] for x in files.values())},
            'selection_policy':'All candidate IDs copied from historical unweighted validation selections, identical across both evaluation weightings; no fresh reselection.',
            'access_policy':'One seed and one system per recipient; A/B same person; raw projection before saved standardizer; no cross-condition auxiliary access.',
            'runtime_seconds':time.perf_counter()-tick,'scientific_fits':0,'optimizer_updates':0}
    atomic_json(out/'FROZEN_OBJECTS.json',result)
    print(json.dumps(result['counts']),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1);run(a.out)
