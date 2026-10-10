"""Receipt-only source-pin builder for independent frozen FIT reader replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ARMS=("O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write_new(path,value,mode):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,mode),"w") as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
    return sha(path)


def build(repo,arm,seed,amendment):
    repo=Path(repo).resolve(strict=True)
    amendment=Path(amendment).resolve(strict=True)
    if arm not in ARMS or seed not in (0,1):raise ValueError("fixed reader source arm/seed differs")
    root=repo/"results/pcrl_incumbent_view_risk_v1"
    r2id=f"R2_{arm}_S{seed}_FIT"
    receipt=root/"r2"/f"{r2id}_UNIT_RECEIPT.json"
    source=json.loads(receipt.read_text())
    if (source.get("source_unit_id")!=r2id or source.get("arm_id")!=arm or
            source.get("source_seed")!=seed or source.get("status") not in (
            "COMPLETE_SELECTED","COMPLETE_IDENTITY_ALIAS","COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK")):
        raise PermissionError("R2 selected reader source incomplete")
    source_map=root/"r1/.private/P0_SOURCE_PATHS.json"
    mapping=json.loads(source_map.read_text())
    private=root/"r2/.private/units"/r2id
    archive_index=private/"PRIVATE_ARCHIVE_INDEX.json"
    reader_archive=private/"READERS.npz"
    fit_law=private/"OSF_DEFENSE_FIT_R0/law.npz"
    fit_sex=Path(mapping["roles"]["OSF_DEFENSE_FIT"]["sex"]["path"])
    bank=Path(mapping["seeds"][str(seed)]["public_bank"]["path"])
    if (sha(fit_sex)!=mapping["roles"]["OSF_DEFENSE_FIT"]["sex"]["sha256"] or
            sha(bank)!=mapping["seeds"][str(seed)]["public_bank"]["sha256"]):
        raise PermissionError("FIT sensitive role/bank source hash differs")
    unit=f"R5_READER_LOSS_REPLAY_{arm}_S{seed}_V1"
    output=root/"r5"/f"{unit}.json"
    config_path=root/"r5/.private"/f"{unit}_CONFIG.json"
    config={"schema":"IVR1-r5-reader-loss-replay-v1", "unit_id":unit,
            "r2_unit_id":r2id,"r2_receipt":str(receipt),
            "private_archive_index":str(archive_index),"readers_archive":str(reader_archive),
            "fit_law":str(fit_law),"fit_sex":str(fit_sex),"public_bank":str(bank),
            "r1_source_map":str(source_map),"output":str(output)}
    config_sha=write_new(config_path,config,0o600)
    venv=Path(os.path.abspath(repo.parents[1]/".venv/bin/python"))
    if not venv.is_file() or not os.access(venv,os.X_OK):
        raise FileNotFoundError("exact scientific venv entrypoint unavailable")
    sources=[config_path,receipt,archive_index,reader_archive,fit_law,fit_sex,bank,
             source_map,amendment,root/"r0/compute_queue.py",
             repo/"incumbent_view/r5_reader_loss_replay.py",
             repo/"incumbent_view/r5_build_reader_loss_specs.py"]
    spec={"unit_id":unit,"stage":"verification","cwd":str(repo),
          "argv":[str(venv),"-m","incumbent_view.r5_reader_loss_replay","--config",str(config_path)],
          "max_wall_s":300,"cpu_reserve_s":300,"max_rss_gib":2.5,
          "projected_output_bytes":100000,"deps":[r2id],"outputs":[str(output)],
          "sources":[{"path":str(p.resolve(strict=True)),"sha256":sha(p)} for p in sources],
          "env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},
          "amendment_sha256":sha(amendment)}
    spec_path=root/"r5/.private"/f"{unit}_QUEUE_SPEC.json"
    spec_sha=write_new(spec_path,spec,0o600)
    public={"schema":"IVR1-r5-reader-loss-spec-index-v1","unit_id":unit,
            "r2_unit_id":r2id,"source_seed":seed,"arm_id":arm,
            "config_sha256":config_sha,"spec_sha256":spec_sha,"source_count":len(sources),
            "amendment_sha256":sha(amendment),"private_spec_relative":str(spec_path.relative_to(repo)),
            "fit_sex_scope_only":True,"attack_or_inner_sex_scope":False,"new_reader_fits":0}
    write_new(root/"r5"/f"{unit}_SPEC_INDEX.json",public,0o644)
    return public


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True)
    p.add_argument("--arm",required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--amendment",required=True);a=p.parse_args()
    print(json.dumps(build(a.repo,a.arm,a.seed,a.amendment),sort_keys=True))


if __name__=="__main__":main()
