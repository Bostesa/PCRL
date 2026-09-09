"""Publish lossless compact completed-unit scores, leaving all person arrays local."""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compact_acs_coalition_strength_evidence import restore

def sha(raw):return hashlib.sha256(raw).hexdigest()

def compact(out):
    matrix=json.loads((out/'EXECUTED_MATRIX.json').read_text());assert len(matrix['systems'])==48
    records={};completed=0
    for e in matrix['systems']:
        if e['reused'] or not e['evaluation_complete']:continue
        completed+=1;base=out/f"seed_{e['seed']}"/e['condition'];cp=base/'complete.json';cb=cp.read_bytes();c=json.loads(cb)
        assert c['evaluation_complete'] and c['condition']==e['condition'] and c['seed']==e['seed']
        for suffix in ('metrics.json','audits/audit_selection.json'):
            p=base/suffix;raw=p.read_bytes();expected=c['files_sha256'][str(p.resolve().relative_to(ROOT))];assert sha(raw)==expected
            stream=io.BytesIO()
            with gzip.GzipFile(filename='',fileobj=stream,mode='wb',compresslevel=9,mtime=0) as z:z.write(raw)
            encoded=stream.getvalue();dest=p.with_suffix('.json.gz')
            if dest.exists():assert dest.read_bytes()==encoded
            else:dest.write_bytes(encoded)
            assert gzip.decompress(encoded)==raw
            records[str(p.relative_to(out))]={'plain_sha256':expected,'gzip_sha256':sha(encoded),'plain_bytes':len(raw),'gzip_bytes':len(encoded),'completion_path':str(cp.relative_to(out)),'completion_sha256':sha(cb),'completion_entry':str(p.resolve().relative_to(ROOT))}
    assert len(records)==2*completed
    manifest={'format':'gzip level9,mtime0,empty filename','original_json_unchanged':True,'scope':f'{completed} completed new interface systems; scores/selection metadata only; no person-level predictions','matrix_status':matrix['status'],'files':records,'source_sha256':sha(Path(__file__).read_bytes())}
    p=out/'COMPACT_UNIT_EVIDENCE.json';encoded=json.dumps(manifest,indent=2)+'\n'
    if p.exists():assert p.read_text()==encoded
    else:p.write_text(encoded)
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--restore',action='store_true');a=ap.parse_args();r=restore(a.out) if a.restore else compact(a.out)
    print(json.dumps({'files':len(r['files']),'plain_bytes':sum(x['plain_bytes'] for x in r['files'].values()),'gzip_bytes':sum(x['gzip_bytes'] for x in r['files'].values()),'all_hashes_verified':True}))
