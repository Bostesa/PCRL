from pathlib import Path
import csv,gzip,json,hashlib,time
from collections import defaultdict
out=Path.cwd()/'results/redesign_20260910_acs_residual_spectral_v1';start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=out/'PER_SEED.csv.gz';table=out/'NEGATIVE_INCREMENT_COUNTS.csv.gz';groups=defaultdict(list)
with gzip.open(source,'rt') as f:
 for r in csv.DictReader(f):
  if r['kind']=='additional_recovery' and r['split']=='test' and r['condition']!='H':
   group='spectral' if r['condition'].startswith('spectral_') else 'historical_augmented'
   groups[group,r['weight'],r['budget'],r['scope']].append(r)
with gzip.open(table,'rt') as f:
 records=list(csv.DictReader(f))
assert len(records)==len(groups)==48
for r in records:
 key=(r['system_group'],r['weight'],r['budget'],r['scope']);values=groups.pop(key)
 numbers=[float(x['value']) for x in values]
 assert int(r['negative_count'])==sum(v<0 for v in numbers)
 assert int(r['below_minus_1e12_count'])==sum(v< -1e-12 for v in numbers)
 assert int(r['total_count'])==len(numbers)
 assert int(r['n_seeds'])==len({v['seed'] for v in values})==3
 assert float(r['minimum_signed_additional_recovery'])==min(numbers)
 endpoints=defaultdict(int)
 for value in values:endpoints[value['endpoint']]+=int(float(value['value'])<0)
 assert json.loads(r['negative_by_endpoint'])==dict(endpoints)
assert not groups
result={'passed':True,'rows':48,'scientific_fits':0,'source_sha256':{str(source):sha(source),str(table):sha(table)},'script_sha256':sha(Path(__file__)),'runtime_seconds':time.perf_counter()-start}
(out/'INDEPENDENT_NEGATIVE_COUNTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
