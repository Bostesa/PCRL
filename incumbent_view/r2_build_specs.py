"""Build owner-only queue specs; preserves the virtualenv executable path.

Configuration construction does not load per-person scientific arrays.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode),"w") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return sha(path)


def interpreter(path):
    p=Path(os.path.abspath(path)) # Do not resolve the venv symlink!
    if not p.is_file() or not os.access(p,os.X_OK):raise ValueError("Python entry path not executable")
    return str(p)


def build_fixture(args):
    repo=Path(args.repo).resolve();root=repo/"results/pcrl_incumbent_view_risk_v1"
    unit="R2_CORRECTNESS_V4";output=root/"r2"/f"{unit}_RECEIPT.json"
    amendment=root/"r0"/args.amendment
    if not (amendment.name.startswith("AMENDMENT_") and "R2_CORRECTNESS_V4" in amendment.name):
        raise ValueError("V4 amendment identity differs")
    files=[*(repo/"incumbent_view"/f"r2_{name}.py" for name in ("tests","space","policy","readers","train","projection","torch_cache","worker","package","materialize","build_specs")),
           repo/"incumbent_view/r1_bank.py",
           repo/"results/pcrl_bank_anchored_release_v1/bank_anchored/r2_laws.py",
           repo/"results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/release.py",
           repo/"results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/bank.py",
           repo/"results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/channel.py",
           repo/"results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/reference_materialize.py",
           root/"r0/compute_queue.py",root/"r0/ADMISSION_LOCK.json",root/"r0/FIXTURE_LOCK.json",
           root/"r0/AMENDMENT_20261010_A_PREFIT_FIXTURES.json",
           root/"r0/AMENDMENT_20261010_C_R2_CORRECTNESS_V3_TIME.json",amendment,
           root/"r2/DESIGN_PRELOCK.md",root/"r2/PREFIT_ENGINEERING_NOTE_V2.md",
           root/"r2/PREFIT_ENGINEERING_NOTE_V3.md",root/"r2/PREFIT_ENGINEERING_NOTE_V4.md",
           root/"r2/R2_CORRECTNESS_V3_RECEIPT.json",
           root/"r2/.private/preserved_failures/R2_CORRECTNESS_V3_RECEIPT_ORIGINAL.json"]
    sources=[{"path":str(p),"sha256":sha(p)} for p in files]
    spec={"unit_id":unit,"stage":"admission","cwd":str(repo),"argv":[interpreter(args.python),"-m","incumbent_view.r2_tests","--unit-id",unit,"--output",str(output)],
          "max_wall_s":120,"cpu_reserve_s":120,"max_rss_gib":1.,"projected_output_bytes":200000,
          "outputs":[str(output)],"deps":[],"sources":sources,"env":{},
          "source_lock_sha256":sha(root/"r0/ADMISSION_LOCK.json"),
          "amendment_sha256":sha(amendment)}
    target=root/"r2/.private/specs"/f"{unit}.json";s=write_json(target,spec)
    public={"schema":"IVR1-r2-fixture-spec-index-v4","unit_id":unit,"spec_sha256":s,
            "sources":[{"path":str(Path(item["path"]).relative_to(repo)),"sha256":item["sha256"]} for item in sources],
            "real_data_reads":False,"queued":False}
    write_json(root/"r2/R2_CORRECTNESS_V4_SPEC_INDEX.json",public,0o644)
    print(json.dumps({"spec":str(target),"sha256":s}))


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--python",required=True)
    p.add_argument("--amendment",required=True)
    p.add_argument("--phase",choices=("fixture",),required=True);args=p.parse_args();build_fixture(args)


if __name__=="__main__":main()
