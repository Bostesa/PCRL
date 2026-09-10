from pathlib import Path
import json,gzip,csv,hashlib,time
out=Path.cwd()/'results/redesign_20260910_acs_residual_spectral_v1';root=Path('/Users/nathansamson/PCRL');start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
refs={};utility={};sources={}
for s in range(3):
 p=root/f'results/redesign_20260908_acs_protection_v1/seed_{s}/metrics.json';sources[str(p)]=sha(p)
 refs[s]={(r['release'],r['target']):r for r in read(p)['raw_metrics'] if r['role']=='transfer' and r['selected']}
 for d in (out/f'seed_{s}').iterdir():
  if (d/'complete.json').exists():
   p=d/'metrics.json';sources[str(p)]=sha(p)
   utility[s,d.name]={r['target']:r for r in read(p)['raw_metrics'] if r['role']=='utility' and 'utility' in r['selected_scopes']}
table=out/'ORIGINAL_CRITERIA.csv.gz';sources[str(table)]=sha(table);n=0;error=0.
with gzip.open(table,'rt') as f:
 for row in csv.DictReader(f):
  s=int(row['seed']);c=row['condition'];key=row['split']+('_person_weighted' if row['weight']=='person_weighted' else '')
  pca={t:refs[s]['E_pca',t][key]['log_loss'] for t in utility[s,c]}
  method={t:r['scores'][key]['log_loss'] for t,r in utility[s,c].items()}
  check=json.loads(row['source_preservation']);passed=True
  for target in ('income_binary','civilian_at_work','public_coverage'):
   original,new=pca[target],method[target];item=check['tasks'][target]
   for field,expected in [('parent_log_loss',original),('erased_log_loss',new),('difference',new-original),('maximum_increase',.01)]:
    difference=abs(item[field]-expected);error=max(error,difference);assert difference<2e-12
   good=new<=original+.01;assert item['pass']==good;passed=passed and good
  assert check['pass']==passed
  reference=min(refs[s][bank,'same_residence'][key]['log_loss'] for bank in ('B_rich_bank','C_tree_bank'))
  parent=pca['same_residence'];new=method['same_residence'];headroom=reference-parent;remaining=reference-new
  retention=json.loads(row['residential_retention']);positive=headroom>0
  assert retention['ratio_defined']==positive and retention['positive_headroom']==positive
  assert retention['pass']==(new<=(reference+parent)/2 if positive else None)
  for field,expected in [('parent_log_loss',parent),('erased_log_loss',new),('reference_log_loss',reference),('parent_headroom',headroom),('erased_headroom',remaining),('erased_minus_parent_log_loss',new-parent)]:
   difference=abs(retention[field]-expected);error=max(error,difference);assert difference<2e-12
  if positive:
   assert abs(retention['retained_fraction']-remaining/headroom)<2e-12
   assert abs(retention['headroom_reduction_fraction']-(1-remaining/headroom))<2e-12
  differences=json.loads(row['task_minus_pca32'])
  assert set(differences)==set(method)
  for t in differences:assert abs(differences[t]-(method[t]-pca[t]))<2e-12
  n+=1
assert n==168
for p,digest in sources.items():assert sha(Path(p))==digest
result={'passed':True,'rows':n,'max_absolute_error':error,'runtime_seconds':time.perf_counter()-start,'source_sha256':sources,'script_sha256':sha(Path(__file__)),'scientific_fits':0}
(out/'INDEPENDENT_LEGACY_CRITERIA.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='source_sha256'})
