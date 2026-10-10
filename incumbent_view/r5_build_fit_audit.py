"""Receipt-only queued spec construction for independent FIT audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ARMS=("O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def fresh(path,value,mode):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700 if mode==0o600 else 0o755)
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,mode),"w") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return sha(path)


def build(repo,arm,seed,amendment):
    repo=Path(repo).resolve(strict=True);amendment=Path(amendment).resolve(strict=True)
    if arm not in ARMS or seed not in (0,1):raise ValueError("fixed arm/seed differs")
    root=repo/"results/pcrl_incumbent_view_risk_v1";unit=f"R2_{arm}_S{seed}_FIT"
    receiptpath=root/"r2"/f"{unit}_UNIT_RECEIPT.json";source=json.loads(receiptpath.read_text())
    if (source.get("source_unit_id")!=unit or source.get("arm_id")!=arm or source.get("source_seed")!=seed
            or source.get("status") not in ("COMPLETE_SELECTED","COMPLETE_IDENTITY_ALIAS","COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK")):
        raise PermissionError("R2 selected FIT receipt unavailable or incomplete")
    source_map_path=root/"r1/.private/P0_SOURCE_PATHS.json";source_map=json.loads(source_map_path.read_text())
    private=root/"r2/.private/units"/unit
    fit=private/"OSF_DEFENSE_FIT_R0/law.npz"
    public_bank=Path(source_map["seeds"][str(seed)]["public_bank"]["path"])
    cross=root/"r2"/f"{unit}_CROSSFIT.json"
    index=private/"PRIVATE_ARCHIVE_INDEX.json";readers=private/"READERS.npz"
    income=Path(source_map["roles"]["OSF_DEFENSE_FIT"]["income"]["path"])
    occupation=Path(source_map["roles"]["OSF_DEFENSE_FIT"]["occupation"]["path"])
    outid=f"R5_FIT_AUDIT_{arm}_S{seed}_V1";r5=root/"r5";out=r5/f"{outid}.json"
    cfgp=r5/".private"/f"{outid}_CONFIG.json"
    cfg={"schema":"IVR1-r5-fit-audit-v1","unit_id":outid,"r2_unit_id":unit,
         "r2_receipt":str(receiptpath),"crossfit":str(cross),"private_archive_index":str(index),
         "readers_archive":str(readers),"fit_law":str(fit),"public_bank":str(public_bank),
         "income":str(income),"occupation":str(occupation),"r1_source_map":str(source_map_path),
         "output":str(out)}
    cfgsha=fresh(cfgp,cfg,0o600)
    pinpaths=[cfgp,receiptpath,cross,index,readers,fit,public_bank,income,occupation,amendment,
              repo/"incumbent_view/r5_fit_audit.py",repo/"incumbent_view/r5_build_fit_audit.py",
              root/"r0/compute_queue.py",source_map_path]
    sources={str(p.resolve(strict=True)):sha(p) for p in pinpaths}
    venv=Path(os.path.abspath(repo.parents[1]/".venv/bin/python"))
    if not venv.is_file() or not os.access(venv,os.X_OK):raise FileNotFoundError("scientific venv entry missing")
    spec={"unit_id":outid,"stage":"verification","cwd":str(repo),
          "argv":[str(venv),"-m","incumbent_view.r5_fit_audit","--config",str(cfgp)],
          "max_wall_s":180,"cpu_reserve_s":180,"max_rss_gib":2.5,"projected_output_bytes":1000000,
          "outputs":[str(out)],"deps":[unit],
          "sources":[{"path":p,"sha256":h} for p,h in sorted(sources.items())],
          "env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},
          "amendment_sha256":sha(amendment)}
    specp=r5/".private"/f"{outid}_QUEUE_SPEC.json";specsha=fresh(specp,spec,0o600)
    indexdoc={"schema":"IVR1-r5-fit-audit-spec-index-v1","unit_id":outid,"r2_unit_id":unit,
              "config_sha256":cfgsha,"spec_sha256":specsha,"source_count":len(sources),
              "amendment_sha256":sha(amendment),"role_arrays_deserialized_by_builder":False,
              "private_spec_relative":str(specp.relative_to(repo))}
    fresh(r5/f"{outid}_SPEC_INDEX.json",indexdoc,0o644)
    return indexdoc


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--arm",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--amendment",required=True);a=p.parse_args()
    print(json.dumps(build(a.repo,a.arm,a.seed,a.amendment),sort_keys=True))


if __name__=="__main__":main()
