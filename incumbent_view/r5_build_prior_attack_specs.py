"""Build two byte-pinned, labels-free historical P* prediction replay units."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_new(path, value, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, mode), "w") as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write("\n")
    return sha(path)


def build(repo, old_repo, amendment, seed):
    repo = Path(repo).resolve(strict=True)
    old_repo = Path(old_repo).resolve(strict=True)
    amendment = Path(amendment).resolve(strict=True)
    root = repo/"results/pcrl_incumbent_view_risk_v1"
    index_path = root/"r1/HISTORICAL_PSTAR_READER_INDEX.json"
    idx = json.loads(index_path.read_text())
    if (idx.get("schema") != "IVR1-historical-pstar-frozen-reader-index-v1"
            or seed not in (0, 1)):
        raise PermissionError("historical index/source identity differs")
    entry = idx["seeds"][str(seed)]
    unit = f"R5_PRIOR_PSTAR_READER_S{seed}_V1"
    source_paths = {"historical_index":index_path}
    for name, item in entry["historical"].items():
        if not isinstance(item, dict) or "alias" not in item:
            continue
        source_paths[name] = old_repo/item["alias"]
    source_paths["p0_inner_release"] = repo/entry["p0_role_only"]["INNER_SELECTION"]["release"]["alias"]
    source_paths["p0_inner_receipt"] = repo/entry["p0_role_only"]["INNER_SELECTION"]["receipt"]["alias"]
    for name, path in source_paths.items():
        if name != "historical_index":
            item = (entry["historical"].get(name) or
                    entry["p0_role_only"]["INNER_SELECTION"].get("release" if name == "p0_inner_release" else "receipt"))
            if sha(path) != item["sha256"]:
                raise PermissionError(f"historical index source bytes differ: {name}")
    output = root/"r5"/f"{unit}.json"
    private = root/"r5/.private"
    config_path = private/f"{unit}_CONFIG.json"
    config = {"schema":"IVR1-r5-historical-pstar-prediction-config-v1",
              "unit_id":unit,"source_seed":seed,"output":str(output),
              **{k:str(v) for k,v in source_paths.items()}}
    config_sha = save_new(config_path, config, 0o600)
    venv = Path(os.path.abspath(repo.parents[1]/".venv/bin/python"))
    if not venv.is_file() or not os.access(venv, os.X_OK):
        raise FileNotFoundError("exact scientific venv entrypoint missing")
    closure = [*source_paths.values(),config_path,amendment,
               root/"r0/compute_queue.py", repo/"incumbent_view/r5_prior_attack_prediction.py",
               repo/"incumbent_view/r5_build_prior_attack_specs.py"]
    sources = [{"path":str(p.resolve(strict=True)),"sha256":sha(p)} for p in closure]
    spec = {"unit_id":unit,"stage":"verification","cwd":str(repo),
            "argv":[str(venv),"-m","incumbent_view.r5_prior_attack_prediction",
                    "--config",str(config_path)],
            "max_wall_s":120,"cpu_reserve_s":120,"max_rss_gib":1.0,
            "projected_output_bytes":50000,"deps":[],"outputs":[str(output)],
            "sources":sources,"env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
                                   "MKL_NUM_THREADS":"1"},
            "amendment_sha256":sha(amendment)}
    spec_path = private/f"{unit}_QUEUE_SPEC.json"
    spec_sha = save_new(spec_path, spec, 0o600)
    public = {"schema":"IVR1-r5-prior-pstar-replay-unit-index-v1","unit_id":unit,
              "source_seed":seed,"source_count":len(sources),"config_sha256":config_sha,
              "spec_sha256":spec_sha,"amendment_sha256":sha(amendment),
              "private_config_relative":str(config_path.relative_to(repo)),
              "private_spec_relative":str(spec_path.relative_to(repo)),
              "new_reader_fits":0,"label_fields_read":[],
              "timing":"AFTER_FIRST_NEW_FIT_STARTED; source prerequisite order deviation"}
    save_new(root/"r5"/f"{unit}_SPEC_INDEX.json", public, 0o644)
    return public


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True)
    p.add_argument("--old-repo",required=True);p.add_argument("--amendment",required=True)
    p.add_argument("--seed",type=int,required=True);a=p.parse_args()
    print(json.dumps(build(a.repo,a.old_repo,a.amendment,a.seed),sort_keys=True))


if __name__=="__main__":
    main()
