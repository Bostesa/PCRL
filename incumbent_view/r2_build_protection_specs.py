"""Prefit source index and post-lock owner-only R2 FIT queue specs.

This builder reads only public metadata and owner-only path/hash manifests. It
never opens role arrays or labels. The fourteen scientific units are generated
only after R0 has sealed and registered CAMPAIGN_LOCK as the protection-stage
amendment; the original ADMISSION_LOCK remains the active queue environment
seal. Do not resolve the Python virtualenv entry symlink.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

ROOT_REL=Path("results/pcrl_incumbent_view_risk_v1")
ARMS=("O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")
ROLES=("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION")
CODE_REL=(
    "incumbent_view/r1_bank.py",
    *(f"incumbent_view/r2_{name}.py" for name in (
        "space","policy","readers","projection","torch_cache","train",
        "package","materialize","worker","tests","build_specs",
        "accounting_tests","build_protection_specs")),
    "results/pcrl_incumbent_view_risk_v1/r0/compute_queue.py",
    "results/pcrl_bank_anchored_release_v1/bank_anchored/r2_laws.py",
    *(f"results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/{name}.py"
      for name in ("release","bank","channel","reference_materialize")),
)


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def fresh_json(path,value,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode),"w") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return sha(path)


def executable_entry(path):
    p=Path(os.path.abspath(path))
    if not p.is_file() or not os.access(p,os.X_OK):raise ValueError("virtualenv Python entry missing")
    if ".venv/bin/python" not in str(p):raise ValueError("expected pinned project virtualenv entry")
    return str(p)


def source_index(repo):
    root=repo/ROOT_REL
    paths=root/"r1/.private/P0_SOURCE_PATHS.json"
    map_=json.loads(paths.read_text())
    if map_.get("schema")!="pcrl-incumbent-r1-private-source-paths-v1":raise ValueError("R1 path map differs")
    sources=[{"relative_path":name,"sha256":sha(repo/name)} for name in CODE_REL]
    admission=root/"r0/ADMISSION_LOCK.json"
    fixture=root/"r2/R2_CORRECTNESS_V4_RECEIPT.json"
    result=json.loads(fixture.read_text())
    if result.get("status")!="PASS" or result.get("tests_run")!=15 or result.get("real_data_reads") is not False:
        raise ValueError("queued V4 correctness gate missing")
    result={"schema":"IVR1-r2-protection-source-index-v2","code":sources,
            "unit_ids":[f"R2_{arm}_S{seed}_FIT" for arm in ARMS for seed in (0,1)],
            "family_order":list(ARMS),"seeds":[0,1],"recipe_seed_rule":"source_seed",
            "fit_role":"OSF_DEFENSE_FIT","read_only_release_roles":list(ROLES[1:]),
            "fit_cpu_ceiling_s":1700,"queue_wall_and_cpu_reserve_s":1800,
            "queue_max_rss_gib":3.0,"admission_lock_sha256":sha(admission),
            "v4_correctness_receipt_sha256":sha(fixture),
            "stream_manifest_sha256":sha(root/"r2/STREAM_ROOT_MANIFEST.json"),
            "owner_only_stream_roots_sha256":sha(root/"r2/.private/stream_roots_v1.json"),
            "owner_only_r1_source_map_sha256":sha(paths),
            "public_bank_sha256_by_seed":{str(s):map_["seeds"][str(s)]["public_bank"]["sha256"] for s in (0,1)},
            "role_release_sha256_by_seed":{str(s):{role:map_["seeds"][str(s)]["roles"][role]["sha256"]
                                                       for role in ROLES} for s in (0,1)}}
    path=root/"r2/PROTECTION_SOURCE_INDEX_V2.json"
    digest=fresh_json(path,result,0o644)
    print(json.dumps({"source_index_relative":str(path.relative_to(repo)),"sha256":digest,
                      "unit_count":len(result["unit_ids"])}))


def _pin(path):return {"path":str(path),"sha256":sha(path)}


def _from_map(rec):
    path=Path(rec["path"])
    if sha(path)!=rec["sha256"]:raise ValueError("R1 mapped source hash differs")
    return {"path":str(path),"sha256":rec["sha256"]}


def build_accounting_fixture(repo,python,amendment):
    root=repo/ROOT_REL;index_path=root/"r2/PROTECTION_SOURCE_INDEX_V2.json"
    index=json.loads(index_path.read_text())
    if index.get("schema")!="IVR1-r2-protection-source-index-v2":raise ValueError("R2 source index differs")
    files=[{"path":str(repo/item["relative_path"]),"sha256":item["sha256"]} for item in index["code"]]
    for item in files:
        if sha(item["path"])!=item["sha256"]:raise ValueError("accounting source changed after index")
    admission=root/"r0/ADMISSION_LOCK.json";amendment_path=root/"r0"/amendment
    if not amendment_path.name.startswith("AMENDMENT_") or amendment_path.suffix!=".json":
        raise ValueError("dated accounting amendment identity differs")
    files.extend(_pin(p) for p in (admission,amendment_path,index_path,
                                    root/"r2/R2_CORRECTNESS_V4_RECEIPT.json"))
    unit="R2_ACCOUNTING_FIXTURE_V1";output=root/"r2"/f"{unit}_RECEIPT.json"
    spec={"unit_id":unit,"stage":"admission","cwd":str(repo),
          "argv":[executable_entry(python),"-m","incumbent_view.r2_accounting_tests",
                  "--unit-id",unit,"--output",str(output)],
          "max_wall_s":120,"cpu_reserve_s":120,"max_rss_gib":1.0,
          "projected_output_bytes":200000,"outputs":[str(output)],"deps":[],
          "sources":files,"env":{},"source_lock_sha256":sha(admission),
          "amendment_sha256":sha(amendment_path)}
    spec_path=root/"r2/.private/specs"/f"{unit}.json";spec_sha=fresh_json(spec_path,spec)
    public={"schema":"IVR1-r2-accounting-fixture-spec-index-v1","unit_id":unit,
            "spec_sha256":spec_sha,"source_index_sha256":sha(index_path),
            "amendment_sha256":sha(amendment_path),"source_count":len(files),"real_data_reads":False}
    public_path=root/"r2/R2_ACCOUNTING_FIXTURE_SPEC_INDEX.json"
    fresh_json(public_path,public,0o644)
    print(json.dumps({"spec_relative":str(spec_path.relative_to(repo)),"sha256":spec_sha}))


def build_specs(repo,python):
    root=repo/ROOT_REL
    index_path=root/"r2/PROTECTION_SOURCE_INDEX_V2.json"
    index=json.loads(index_path.read_text())
    if index.get("schema")!="IVR1-r2-protection-source-index-v2":raise ValueError("R2 source index differs")
    if tuple(index["unit_ids"])!=tuple(f"R2_{arm}_S{seed}_FIT" for arm in ARMS for seed in (0,1)):
        raise ValueError("R2 fixed unit registry differs")
    for item in index["code"]:
        if sha(repo/item["relative_path"])!=item["sha256"]:raise ValueError("science code changed after prelock index")
    admission=root/"r0/ADMISSION_LOCK.json";science=root/"r0/CAMPAIGN_LOCK.json"
    admission_sha=sha(admission);science_sha=sha(science)
    if admission_sha!=index["admission_lock_sha256"]:raise ValueError("admission seal changed")
    queue=json.loads((root/"r0/.private/compute/state.json").read_text())
    if queue["seal"]["sha256"]!=admission_sha:raise ValueError("queue admission seal differs")
    amendments=[item["sha256"] for item in queue["amendments"] if "protection" in item["stages"]]
    if not amendments or science_sha not in amendments:raise ValueError("campaign lock not registered protection amendment")
    operational_amendment_sha=amendments[-1]
    v4=root/"r2/R2_CORRECTNESS_V4_RECEIPT.json"
    if sha(v4)!=index["v4_correctness_receipt_sha256"]:raise ValueError("correctness gate changed")
    accounting=root/"r2/R2_ACCOUNTING_FIXTURE_V1_RECEIPT.json"
    proof=json.loads(accounting.read_text())
    if proof.get("status")!="PASS" or proof.get("tests_run")!=3 or proof.get("real_data_reads") is not False:
        raise ValueError("accounting correctness gate missing")
    source_map=root/"r1/.private/P0_SOURCE_PATHS.json"
    if sha(source_map)!=index["owner_only_r1_source_map_sha256"]:raise ValueError("R1 source map changed")
    mapped=json.loads(source_map.read_text())
    stream=root/"r2/STREAM_ROOT_MANIFEST.json";roots=root/"r2/.private/stream_roots_v1.json"
    if sha(stream)!=index["stream_manifest_sha256"] or sha(roots)!=index["owner_only_stream_roots_sha256"]:
        raise ValueError("frozen sampling roots differ")
    py=executable_entry(python)
    output=[]
    for arm in ARMS:
        for seed in (0,1):
            unit=f"R2_{arm}_S{seed}_FIT"
            z=mapped["seeds"][str(seed)]
            bank=_from_map(z["public_bank"])
            if bank["sha256"]!=index["public_bank_sha256_by_seed"][str(seed)]:raise ValueError("public bank index differs")
            labels={k:_from_map(mapped["roles"]["OSF_DEFENSE_FIT"][k]) for k in ("income","occupation","sex")}
            roles={}
            for role in ROLES:
                record=z["roles"][role]
                release=_from_map(record)
                receipt={"path":record["receipt_path"],"sha256":record["receipt_sha256"]}
                if sha(receipt["path"])!=receipt["sha256"]:raise ValueError("R1 role receipt differs")
                if release["sha256"]!=index["role_release_sha256_by_seed"][str(seed)][role]:
                    raise ValueError("P0 role release index differs")
                roles[role]={"release":release,"receipt":receipt}
            pins={"admission_lock":_pin(admission),"science_lock":_pin(science),
                  "stream_roots":_pin(roots),"stream_manifest":_pin(stream),"public_bank":bank}
            private=root/"r2/.private/units"/unit
            cfg={"schema":"IVR1-r2-fit-config-v1","unit_id":unit,"arm_id":arm,
                 "source_seed":seed,"recipe_seed":seed,"source_lock_sha256":science_sha,
                 "science_lock_sha256":science_sha,"admission_lock_sha256":admission_sha,
                 "protection_amendment_sha256":operational_amendment_sha,
                 "source_binding_sha256":json.loads(Path(bank["path"]).read_text())["source_binding_sha256"],
                 "stream_roots_sha256":sha(roots),"stream_manifest_sha256":sha(stream),
                 "pins":pins,"role_p0":roles,"fit_labels":labels,
                 "private_output":str(private),"public_output":str(root/"r2"),
                 "fit_cpu_ceiling_s":index["fit_cpu_ceiling_s"]}
            cfg_path=root/"r2/.private/configs"/f"{unit}.json";cfg_sha=fresh_json(cfg_path,cfg)
            sources=[{"path":str(repo/item["relative_path"]),"sha256":item["sha256"]} for item in index["code"]]
            sources.extend(_pin(p) for p in (index_path,v4,accounting,source_map))
            sources.extend(pins.values())
            sources.extend(labels.values())
            for item in roles.values():sources.extend(item.values())
            sources.append({"path":str(cfg_path),"sha256":cfg_sha})
            seen=set();sources=[item for item in sources if not (item["path"] in seen or seen.add(item["path"]))]
            spec={"unit_id":unit,"stage":"protection","cwd":str(repo),
                  "argv":[py,"-m","incumbent_view.r2_worker","--config",str(cfg_path)],
                  "max_wall_s":index["queue_wall_and_cpu_reserve_s"],
                  "cpu_reserve_s":index["queue_wall_and_cpu_reserve_s"],
                  "max_rss_gib":index["queue_max_rss_gib"],"projected_output_bytes":150000000,
                  "outputs":[str(root/"r2"/f"{unit}_UNIT_RECEIPT.json"),
                             str(root/"r2"/f"{unit}_CROSSFIT.json")],
                  "deps":[],"sources":sources,"env":{},
                  "source_lock_sha256":science_sha,"admission_lock_sha256":admission_sha,
                  "amendment_sha256":operational_amendment_sha}
            spec_path=root/"r2/.private/specs"/f"{unit}.json";spec_sha=fresh_json(spec_path,spec)
            output.append({"unit_id":unit,"config_sha256":cfg_sha,"spec_sha256":spec_sha,
                           "sources":len(sources),"arm_id":arm,"source_seed":seed})
    public={"schema":"IVR1-r2-protection-spec-index-v1","source_index_sha256":sha(index_path),
            "science_lock_sha256":science_sha,"admission_lock_sha256":admission_sha,
            "protection_amendment_sha256":operational_amendment_sha,
            "units":output,"venv_entry_suffix":".venv/bin/python"}
    public_path=root/"r2/PROTECTION_SPEC_INDEX.json";digest=fresh_json(public_path,public,0o644)
    print(json.dumps({"spec_index_relative":str(public_path.relative_to(repo)),"sha256":digest,
                      "unit_count":len(output)}))


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("index","accounting","specs"),required=True)
    p.add_argument("--repo",required=True);p.add_argument("--python");p.add_argument("--amendment")
    a=p.parse_args();repo=Path(a.repo).resolve(strict=True)
    if a.phase=="index":source_index(repo)
    elif a.phase=="accounting":
        if not a.python or not a.amendment:raise ValueError("--python and --amendment required")
        build_accounting_fixture(repo,a.python,a.amendment)
    else:
        if not a.python:raise ValueError("--python required after science seal")
        build_specs(repo,a.python)


if __name__=="__main__":main()
