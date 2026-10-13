"""Predeclared regression for mutable queue logs in custody plans."""
import argparse,json,os,subprocess
from pathlib import Path
from .scheduler import assert_active_from_env
from .source import write_once,sha_file
from .custody import create_plan


def run(private_output,public_output):
    active=assert_active_from_env(stage='custody');root=Path(private_output)
    root.mkdir(parents=True,exist_ok=False,mode=0o700);repo=root/'fixture_repo';repo.mkdir(mode=0o700)
    env=dict(os.environ);env.update(GIT_AUTHOR_NAME='PCRL fixture',GIT_COMMITTER_NAME='PCRL fixture',GIT_AUTHOR_EMAIL='fixture@example.invalid',GIT_COMMITTER_EMAIL='fixture@example.invalid')
    def git(*args):return subprocess.run(['git',*args],cwd=repo,env=env,check=True,capture_output=True)
    git('init');(repo/'fixture.txt').write_text('harmless\n');git('add','fixture.txt');git('commit','-m','Fixture')
    units=repo/'.private/conditional_release/compute/units'
    for name in (active['unit_id'],'R1_FULL_SOURCE_ADMISSION_V1','UNREGISTERED_FIXTURE_UNIT'):
        d=units/name;d.mkdir(parents=True);(d/'stdout.log').write_text('synthetic log only\n')
    manifest=root/'SOURCE_PATHS.json';write_once(manifest,{'schema':'CBR1-role-safe-private-source-v1'})
    plan=root/'PLAN.json';create_plan(repo,manifest,plan,root/'PLAN_RECEIPT.json')
    aliases={x['alias'] for x in json.loads(plan.read_text())['entries']}
    live=f"campaign/compute/units/{active['unit_id']}/stdout.log";unknown='campaign/compute/units/UNREGISTERED_FIXTURE_UNIT/stdout.log';done='campaign/compute/units/R1_FULL_SOURCE_ADMISSION_V1/stdout.log'
    checks={'own_running_unit_excluded':live not in aliases,'unregistered_unit_excluded':unknown not in aliases,'terminal_unit_included':done in aliases}
    write_once(public_output,{'schema':'CBR1-custody-mutable-log-regression-v1','unit_id':active['unit_id'],'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'fixture_only':True,'scientific_arrays_read':False},False)
    if not all(checks.values()):raise AssertionError('custody plan includes mutable or unregistered unit logs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-output',required=True);p.add_argument('--public-output',required=True);a=p.parse_args();run(a.private_output,a.public_output)
