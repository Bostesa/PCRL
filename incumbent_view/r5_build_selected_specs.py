"""Receipt-only builder for independent selected-law utility replay specs.

The builder hashes protected files as opaque bytes. It never deserializes role
arrays or labels. R0 must register a dated verification amendment and enqueue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ARMS = ("O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_once(path, data, mode):
    path = Path(path)
    path.parent.mkdir(mode=0o700 if mode == 0o600 else 0o755, parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, mode), "w") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False);f.write("\n")
    return sha(path)


def build(repo, arm, seed, role, amendment):
    repo = Path(repo).resolve(strict=True)
    amendment = Path(amendment).resolve(strict=True)
    if arm not in ARMS or seed not in (0, 1) or role not in ("FIT", "INNER"):
        raise ValueError("selected utility replay identity differs")
    root = repo/"results/pcrl_incumbent_view_risk_v1"
    r3id = f"R3_{arm}_S{seed}_{role}_PROTECTED_V1"
    r5id = f"R5_SELECTED_UTILITY_REPLAY_{arm}_S{seed}_{role}_V1"
    r3config = root/"r3/.private"/f"{r3id}_CONFIG.json"
    r3csv = root/"r3"/f"{r3id}.csv"
    r3receipt = root/"r3"/f"{r3id}_RECEIPT.json"
    original = json.loads(r3config.read_text())
    receipt = json.loads(r3receipt.read_text())
    if (original.get("unit_id") != r3id or original.get("arm_id") != arm or
            original.get("source_seed") != seed or original.get("role") !=
            ("OSF_DEFENSE_FIT" if role == "FIT" else "INNER_SELECTION") or
            receipt.get("unit_id") != r3id or receipt.get("csv_sha256") != sha(r3csv)):
        raise PermissionError("R3 selected scorer source identity differs")
    venv = Path(os.path.abspath(repo.parents[1]/".venv/bin/python"))
    if not venv.is_file() or not os.access(venv, os.X_OK):
        raise FileNotFoundError("exact scientific venv entrypoint unavailable")
    r5 = root/"r5"
    out = r5/f"{r5id}.json"
    cfgpath = r5/".private"/f"{r5id}_CONFIG.json"
    cfg = {"schema":"IVR1-r5-selected-utility-replay-v1", "unit_id":r5id,
           "r3_config":str(r3config),"r3_csv":str(r3csv),"r3_receipt":str(r3receipt),
           "output":str(out)}
    cfgsha = write_once(cfgpath, cfg, 0o600)
    pinpaths = [cfgpath,r3config,r3csv,r3receipt,amendment,
                repo/"incumbent_view/r5_selected_utility.py",
                repo/"incumbent_view/r5_build_selected_specs.py",
                root/"r0/compute_queue.py",
                *(Path(v["path"]) for v in original["pins"].values())]
    sources = {str(p.resolve(strict=True)):sha(p) for p in pinpaths}
    spec = {"unit_id":r5id,"stage":"verification","cwd":str(repo),
            "argv":[str(venv),"-m","incumbent_view.r5_selected_utility","--config",str(cfgpath)],
            "max_wall_s":300,"cpu_reserve_s":300,"max_rss_gib":2.5,"projected_output_bytes":100000,
            "deps":[r3id],"outputs":[str(out)],
            "sources":[{"path":p,"sha256":h} for p,h in sorted(sources.items())],
            "env":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},
            "amendment_sha256":sha(amendment)}
    specpath = r5/".private"/f"{r5id}_QUEUE_SPEC.json"
    specsha = write_once(specpath, spec, 0o600)
    index = {"schema":"IVR1-r5-selected-utility-spec-index-v1","unit_id":r5id,
             "r3_unit_id":r3id,"config_sha256":cfgsha,"spec_sha256":specsha,
             "source_count":len(sources),"amendment_sha256":sha(amendment),
             "private_config_relative":str(cfgpath.relative_to(repo)),
             "private_spec_relative":str(specpath.relative_to(repo)),
             "role_arrays_deserialized_by_builder":False}
    indexpath = r5/f"{r5id}_SPEC_INDEX.json"
    write_once(indexpath,index,0o644)
    return index


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--arm",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--role",required=True)
    p.add_argument("--amendment",required=True);a=p.parse_args()
    print(json.dumps(build(a.repo,a.arm,a.seed,a.role,a.amendment),sort_keys=True))


if __name__=="__main__":
    main()
