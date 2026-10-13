"""Queue-only synthetic custody engineering: no source person arrays."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
from .scheduler import assert_active_from_env
from .source import write_once,sha_file
from .custody import create_plan,snapshot,restore,copy_pinned,safe_alias,uncached_hash


def run(private_output,public_output):
    active=assert_active_from_env(stage='custody')
    root=Path(private_output)
    if root.exists():raise FileExistsError('new fixture version required')
    root.mkdir(parents=True,mode=0o700);repo=root/'engineering_repo';repo.mkdir(mode=0o700)
    env=dict(os.environ);env['GIT_AUTHOR_NAME']=env['GIT_COMMITTER_NAME']='PCRL custody fixture';env['GIT_AUTHOR_EMAIL']=env['GIT_COMMITTER_EMAIL']='fixture@example.invalid'
    def git(*args):return subprocess.run(['git',*args],cwd=repo,env=env,check=True,capture_output=True,text=True)
    git('init');payload=repo/'fixture.txt';payload.write_bytes(b'role-safe opaque fixture\n');git('add','fixture.txt');git('commit','-m','Synthetic custody fixture')
    source=root/'source.txt';source.write_bytes(b'known harmless bytes\n');source.chmod(0o600)
    manifest=root/'SOURCE_PATHS.json';write_once(manifest,{'schema':'CBR1-role-safe-private-source-v1','fixture':{'path':str(source),'sha256':sha_file(source)}})
    private=repo/'.private/conditional_release/r1';private.mkdir(parents=True,mode=0o700);(private/'private.txt').write_bytes(b'private fixture only\n')
    plan=root/'PLAN.json';p=create_plan(repo,manifest,plan,root/'PLAN_RECEIPT.json')
    snap=root/'snapshot';s=snapshot(repo,plan,p['plan_sha256'],snap,root/'SNAPSHOT_RECEIPT.json')
    target=root/'restore';t=restore(snap,target,root/'RESTORE_RECEIPT.json')
    remapped=json.loads((target/'runtime/SOURCE_PATHS.json').read_text())
    if remapped['fixture']['path']==str(source) or Path(remapped['fixture']['path']).read_bytes()!=source.read_bytes():raise AssertionError('restore depends on original source')
    if uncached_hash(target/'execution.gitbundle')!=s['bundle_sha256']:raise AssertionError('bundle restore differs')
    refused=[]
    for alias in ('../escape','/absolute','x\\y','a/../escape'):
        try:safe_alias(alias)
        except PermissionError:refused.append(alias)
        else:raise AssertionError('unsafe alias accepted')
    bad=root/'changed.txt';bad.write_bytes(b'changed')
    try:copy_pinned(bad,root/'should_not_exist.txt','0'*64,bad.stat().st_size)
    except PermissionError:pass
    else:raise AssertionError('changed input copied')
    report={'schema':'CBR1-custody-engineering-fixture-v1','unit_id':active['unit_id'],'status':'PASS','fixture_only':True,'scientific_arrays_read':False,'actual_git_bundle_verify':True,'uncached_snapshot_and_restore_hashes':True,'restored_source_pin_remapped':True,'unsafe_aliases_refused':len(refused),'changed_source_refused':True,'file_count':t['file_count']}
    write_once(public_output,report,False);print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-output',required=True);p.add_argument('--public-output',required=True);a=p.parse_args();run(a.private_output,a.public_output)
