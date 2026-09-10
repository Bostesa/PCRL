"""Lossless compact evidence export/restore; never packages person arrays or fitted objects."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(out,restore=False):
    if restore:
        manifest=json.loads((out/'EXPORT_MANIFEST.json').read_text())
        for r in manifest['exports']:
            packed=out/r['packed'];raw=out/r['original'];data=packed.read_bytes();assert sha(data)==r['packed_sha256'];plain=gzip.decompress(data);assert sha(plain)==r['original_sha256']
            if raw.exists():assert sha(raw.read_bytes())==r['original_sha256'],str(raw)
            else:raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(plain)
        print('Verified/restored',len(manifest['exports']),'lossless evidence files; no fitted objects restored.');return
    tick=time.perf_counter();omitted=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and (p.suffix in ('.pt','.joblib','.npz','.npy','.log') or 'fitted' in p.parts or p.name.endswith('.local.json')):
            data=p.read_bytes();omitted.append({'path':str(p.relative_to(ROOT)),'bytes':len(data),'sha256':sha(data)})
    write(out/'LOCAL_ARTIFACTS.json',{'root':str(ROOT),'omitted_files':omitted,'total_bytes':sum(x['bytes'] for x in omitted),'historical_dependencies':'REUSE_MANIFEST.json and PREFIT_IDENTITY.json retain original paths/hashes; no duplicate archive'})
    excluded={'PROTOCOL.md','PREFIT_REVIEW.md','config.json','comparison_rules.json','INPUT_SCHEMA.json','protocol_freeze.json','PREFIT_IDENTITY.json','ANCHOR_PARITY.json','REUSE_MANIFEST.json'}
    exports=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.suffix not in ('.csv','.json') or 'fitted' in p.parts or p.name.endswith('.local.json') or p.name in excluded or p.stat().st_size<250000:continue
        data=p.read_bytes();packed=gzip.compress(data,compresslevel=9,mtime=0);q=p.with_suffix(p.suffix+'.gz')
        if q.exists():assert q.read_bytes()==packed,'Existing incompatible export '+str(q)
        else:q.write_bytes(packed)
        assert gzip.decompress(q.read_bytes())==data
        exports.append({'original':str(p.relative_to(out)),'packed':str(q.relative_to(out)),'original_bytes':len(data),'packed_bytes':len(packed),'original_sha256':sha(data),'packed_sha256':sha(packed)})
    write(out/'EXPORT_MANIFEST.json',{'codec':'gzip, mtime0, lossless exact bytes','exports':exports,'original_bytes':sum(x['original_bytes'] for x in exports),'packed_bytes':sum(x['packed_bytes'] for x in exports),'runtime_seconds':time.perf_counter()-tick})
    # Original files stay local for hash-bound replay; Git publishes only one compact copy.
    ignored=['*.local.json','*.pt','*.npz','*.npy','*.joblib','*.log','fitted/','logs/']+['/'+x['original'] for x in exports]
    (out/'.gitignore').write_text('\n'.join(ignored)+'\n')
    for p in out.glob('*.md'):
        if p.name in excluded:continue
        s=p.read_text()
        for x in exports:s=s.replace(']('+x['original']+')',']('+x['packed']+')')
        p.write_text(s)
    print(json.dumps({'exports':len(exports),'packed_bytes':sum(x['packed_bytes'] for x in exports),'omitted_files':len(omitted),'runtime_seconds':time.perf_counter()-tick},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--restore',action='store_true');a=p.parse_args();main(a.out.resolve(),a.restore)
