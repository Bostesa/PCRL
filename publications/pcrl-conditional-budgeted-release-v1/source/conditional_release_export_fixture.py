"""Run all campaign and independent synthetic fixtures from the export tree."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
from conditional_release.scheduler import assert_active_from_env


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--unit-id',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--private-junit',required=True)
    parser.add_argument('--public-output',required=True)
    args=parser.parse_args()
    assert_active_from_env(args.unit_id,stage='verification')
    root=Path(__file__).resolve().parent
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(args.manifest)!=args.manifest_sha256:raise PermissionError('export manifest changed')
    manifest=json.loads(Path(args.manifest).read_text())
    for item in manifest['files']:
        p=(root/item['path']).resolve()
        if not p.is_relative_to(root) or digest(p)!=item['sha256']:raise PermissionError('export source changed')
    junit=Path(args.private_junit);junit.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if junit.exists() or Path(args.public_output).exists():raise FileExistsError('new fixture outputs required')
    argv=[sys.executable,'-m','pytest','-q','conditional_release','conditional_release_independent','--junitxml',str(junit)]
    env=dict(os.environ);env.update({'PYTHONPATH':str(root),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','BLIS_NUM_THREADS':'1'})
    started=time.monotonic();result=subprocess.run(argv,cwd=root,env=env,text=True,capture_output=True)
    print(result.stdout);print(result.stderr,file=sys.stderr)
    if junit.exists():
        suites=ET.parse(junit).getroot();counts={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    else:counts={k:0 for k in ('tests','failures','errors','skipped')}
    receipt={'schema':'CBR1-public-export-full-fixture-receipt-v1','unit_id':args.unit_id,'status':'PASS' if result.returncode==0 and counts['tests']>0 and counts['failures']==counts['errors']==0 else 'FAIL','exit_code':result.returncode,'counts':counts,'elapsed_s':time.monotonic()-started,'export_manifest_sha256':args.manifest_sha256,'source_file_count':len(manifest['files']),'executed_command':['PYTHON','-m','pytest','-q','conditional_release','conditional_release_independent','--junitxml','PRIVATE_JUNIT_XML'],'test_exclusions':[],'scope':'All campaign and independent synthetic fixtures; no person arrays or fitted candidate endpoints opened','numerical_threads':1}
    target=Path(args.public_output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    raise SystemExit(0 if receipt['status']=='PASS' else 1)


if __name__=='__main__':main()
