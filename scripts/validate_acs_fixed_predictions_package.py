"""Read-only publication checks, independent of scientific fitting."""
import datetime,gzip,hashlib,json,py_compile,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 tick=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc)
 if not (OUT/'PACKAGING_VALIDATION.json').exists():(OUT/'PACKAGING_VALIDATION.json').write_text('{"status":"checking"}\n')
 from experiments.run_acs_fixed_predictions import verify
 verify(OUT)
 source=list(ROOT.glob('experiments/*fixed_predictions*.py'))+list(ROOT.glob('scripts/*fixed_predictions*.py'))+[ROOT/'tests/test_acs_fixed_predictions.py']
 for p in source:py_compile.compile(str(p),doraise=True)
 export=json.loads((OUT/'EXPORT_MANIFEST.json').read_text());checks=0
 for r in export['exports']:
  packed=OUT/r['packed'];data=gzip.decompress(packed.read_bytes());assert sha(packed)==r['packed_sha256'] and hashlib.sha256(data).hexdigest()==r['original_sha256'];assert data==(OUT/r['original']).read_bytes();checks+=1
 ignored=set(subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard',str(OUT.relative_to(ROOT))],cwd=ROOT,text=True).splitlines())
 eligible=subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(OUT.relative_to(ROOT))],cwd=ROOT,text=True).splitlines()
 for name in eligible:
  p=ROOT/name;assert p.suffix not in ('.pt','.npy','.npz','.joblib','.log') and 'fitted' not in p.parts and not p.name.endswith('.local.json'),name
 for r in export['exports']:assert str((OUT/r['original']).relative_to(ROOT)) in ignored
 links=[]
 for p in OUT.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if target.startswith(('https:','http:','#')):continue
   q=(p.parent/target.split('#')[0]).resolve();assert q.exists(),(p,target)
   rel=str(q.relative_to(ROOT))
   assert rel not in ignored,(p,'published link points to ignored file',target)
   links.append((str(p.relative_to(ROOT)),target))
 matrix=json.loads((OUT/'EXECUTED_MATRIX.json').read_text());assert matrix['complete'] and matrix['evaluated_systems']==18 and matrix['learned_continuations']==12
 replay=json.loads((OUT/'COMPARISON_REPLAY.json').read_text());assert replay['passed']
 # Check new full metric aggregates contain no person-level vector payloads.
 forbidden={'raw_rows','row_indices','raw_predictions','probabilities','person_ids','household_ids'}
 def scan(x):
  if isinstance(x,dict):
   assert not (forbidden&set(x)),forbidden&set(x)
   for v in x.values():scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
 for p in OUT.glob('seed_*/*/metrics.json'):scan(json.loads(p.read_text()))
 state=json.loads((OUT/'START_STATE.local.json').read_text())
 result={'passed':True,'started_utc':start.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-tick,'compiled_modules':len(source),'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},'frozen_scientific_closure_unchanged':True,'lossless_export_checks':checks,'local_link_checks':len(links),'publishable_untracked_study_files':len(eligible),'no_fitted_or_person_array_files':True,'full_matrix_complete':True,'compact_metrics_payload_checked':True,'starting_state_saved_locally':True}
 (OUT/'PACKAGING_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()
