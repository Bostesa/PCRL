"""Receipt-only two-phase final-attack spec builder; no array deserialization.

R0 supplies an owner-only input map of exact selected R2 units and all R3
scorer receipts. Prepare preserves all fixed slots and hashes worker specs;
R0 then seals/registers those exact hashes. Finalize adds the mandatory latest
attack amendment SHA to each queue spec and never mutates prepared specs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from .r4_worker import ARMS, MODULES, validate_gate

STOCHASTIC=("O_J","V_J","V_P","V_L","V_S12","V_S21")
ROLES=("ATTACK_FIT_NEW","INNER_SELECTION")
THREAD_ENV={k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")}


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def pin(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise PermissionError("required receipt/source missing or symlink")
    return {"path":str(path.absolute()),"sha256":sha(path)}


def entry(pinvalue):
    if set(pinvalue)!={"path","sha256"} or pin(pinvalue["path"])["sha256"]!=pinvalue["sha256"]:
        raise PermissionError("owner source map does not match source bytes")
    return pinvalue


def new_json(path,payload,private=True):
    path=Path(path)
    if path.exists():raise FileExistsError("prepared owner artifact is write-once")
    path.parent.mkdir(parents=True,mode=0o700 if private else 0o755,exist_ok=True)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644),"w") as f:
        json.dump(payload,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return sha(path)


def _module_pins(repo):
    result=[]
    for module,relative in MODULES.items():
        value=pin(repo/relative);value["module"]=module;result.append(value)
    return result


def _source_list(spec):
    values=[]
    def add(item):
        if item is None:return
        if type(item)==dict:
            if "path" in item and "sha256" in item:values.append({"path":item["path"],"sha256":item["sha256"]})
            else:
                for child in item.values():add(child)
        elif type(item)==list:
            for child in item:add(child)
    for key in ("campaign_seal","role_manifest","public_bank","eligibility","source_pins",
                "utility_links","role_releases","sex_archives","policy_package"):
        add(spec[key])
    unique={}
    for value in values:
        path=Path(value["path"]).absolute()
        if path in unique and unique[path]!=value["sha256"]:raise PermissionError("same source has conflicting hashes")
        unique[path]=value["sha256"]
    return [{"path":str(path),"sha256":digest} for path,digest in sorted(unique.items())]


def prepare(config_path):
    config_path=Path(config_path).absolute();cfg=json.loads(config_path.read_text())
    if cfg.get("schema")!="pcrl-incumbent-r4-owner-plan-v1" or set(cfg["arms"])!=set(ARMS):
        raise ValueError("all eight prospective arms required")
    repo=Path(cfg["repo"]).resolve(strict=True)
    root=repo/"results/pcrl_incumbent_view_risk_v1/r4"
    private=root/".private";private.mkdir(mode=0o700,exist_ok=True)
    if private.stat().st_mode&0o077:raise PermissionError("owner output root loose")
    python=Path(os.path.abspath(cfg["python_executable"])) # retain venv symlink entrypoint
    if not python.is_file() or not os.access(python,os.X_OK):raise ValueError("venv executable missing")
    campaign=entry(cfg["campaign_seal"]);manifest=entry(cfg["role_manifest"])
    cfgpin=pin(config_path);modules=_module_pins(repo)
    unit_index={};slots=[]
    for arm in ARMS:
        record=cfg["arms"][arm]
        gate_item=entry(record["gate"]) if record.get("gate") else None
        gate=json.loads(Path(gate_item["path"]).read_text()) if gate_item else None
        if gate is not None and gate.get("all_two_seed_task_reference_point_caps_pass") is True:
            validate_gate(gate,{"arm_id":arm});status="ATTACK_PLANNED"
        else:
            status="POINT_INELIGIBLE" if gate else "INCOMPLETE_SEARCH"
        for seed in (0,1):
            bank=entry(cfg["seeds"][str(seed)]["public_bank"])
            sex={role:entry(item) for role,item in cfg["seeds"][str(seed)]["sex_archives"].items()}
            if set(sex)!=set(ROLES):raise ValueError("both final label roles required")
            draws=range(3) if arm in STOCHASTIC else range(1)
            for draw in draws:
                uid=f"R4_FINAL_{arm}_S{seed}_R{draw}_V1"
                slots.append({"unit_id":uid,"arm_id":arm,"source_seed":seed,"realization":draw,"status":status})
                if status!="ATTACK_PLANNED":continue
                r2=record["seeds"][str(seed)]
                releases={}
                for role in ROLES:
                    item=r2["releases"][role][str(draw)]
                    releases[role]={key:entry(value) for key,value in item.items()}
                    if set(releases[role])!=({"codes","receipt"} if arm=="P0" else {"codes","receipt","law"}):
                        raise ValueError("selected role release pin set differs")
                links={}
                for key in ("s0_fit","s0_inner","s1_fit","s1_inner"):
                    item=record["utility_links"][key]
                    links[key]={name:(None if value is None else entry(value)) for name,value in item.items()}
                    if set(links[key])!=({"receipt","csv","law","law_receipt","policy_receipt"}):
                        raise ValueError("utility/scorer link set differs")
                    source=gate["source_receipts"][key]
                    if (source["receipt_sha256"]!=links[key]["receipt"]["sha256"] or
                            source["csv_sha256"]!=links[key]["csv"]["sha256"]):
                        raise PermissionError("gate/scorer source receipt differs")
                policy=None if arm=="P0" else {name:entry(value) for name,value in r2["policy_package"].items()}
                if arm!="P0" and set(policy)!={"receipt","archive"}:
                    raise ValueError("selected policy package incomplete")
                worker={"schema":"pcrl-incumbent-r4-final-unit-v1","unit_id":uid,"arm_id":arm,
                        "source_seed":seed,"realization":draw,"attacker_seed":100000+1000*seed+draw,
                        "campaign_seal":campaign,"role_manifest":manifest,"public_bank":bank,"eligibility":gate_item,
                        "source_pins":modules,"utility_links":links,"role_releases":releases,
                        "sex_archives":sex,"policy_package":policy,
                        "private_output_path":str(private/"attacks"/uid),
                        "public_receipt_path":str(root/f"{uid}_RECEIPT.json")}
                spec_path=private/"worker_specs"/f"{uid}.json";spec_sha=new_json(spec_path,worker)
                unit_index[uid]={"spec_path":str(spec_path),"spec_sha256":spec_sha,
                                 "arm_id":arm,"source_seed":seed,"realization":draw,
                                 "source_count":len(_source_list(worker))}
    public={"schema":"pcrl-incumbent-r4-prepared-index-v1","status":"RECEIPT_ONLY_NO_SEX_READ",
            "config_sha256":cfgpin["sha256"],"campaign_seal_sha256":campaign["sha256"],
            "all_slots":slots,"prepared_units":unit_index,"planned_fits":30*len(unit_index),
            "unit_count":len(unit_index),"aliases":[],"source_worker":"incumbent_view.r4_worker"}
    index_path=root/"ATTACK_PREPARED_INDEX.json";digest=new_json(index_path,public,private=False)
    return {"index_path":str(index_path),"index_sha256":digest,"unit_count":len(unit_index),"slot_count":len(slots)}


def finalize(index_path,lock_path,lock_sha256,python,repo):
    index=json.loads(Path(index_path).read_text());lockpath=Path(lock_path)
    if sha(lockpath)!=lock_sha256:raise PermissionError("R0 registered attack lock bytes differ")
    lock=json.loads(lockpath.read_text())
    if (lock.get("schema")!="pcrl-incumbent-final-attack-stage-lock-v1" or
            lock.get("status")!="LOCALLY_SEALED_ATTACKS" or
            set(lock.get("units",{}))!=set(index["prepared_units"])):
        raise PermissionError("R0 attack-stage lock did not bind exact prepared units")
    repo=Path(repo).resolve(strict=True);root=repo/"results/pcrl_incumbent_view_risk_v1/r4"
    python=Path(os.path.abspath(python))
    if not python.is_file() or not os.access(python,os.X_OK):raise ValueError("venv entrypoint missing")
    queue={}
    for uid,entrydata in index["prepared_units"].items():
        prepared=entry(entrydata["spec_path"] if isinstance(entrydata["spec_path"],dict) else
                       {"path":entrydata["spec_path"],"sha256":entrydata["spec_sha256"]})
        unit=json.loads(Path(prepared["path"]).read_text())
        expected={"spec_sha256":entrydata["spec_sha256"],"arm_id":unit["arm_id"],
                  "source_seed":unit["source_seed"],"realization":unit["realization"],"point_eligible":True}
        if lock["units"].get(uid)!=expected:raise PermissionError("exact unit missing from stage lock")
        sources=_source_list(unit)+[pin(lockpath),prepared]
        sources={v["path"]:v for v in sources}
        spec={"unit_id":uid,"stage":"attacks","argv":[str(python),"-m","incumbent_view.r4_worker",
              "--spec",prepared["path"],"--spec-sha256",prepared["sha256"],
              "--stage-lock",str(lockpath),"--stage-lock-sha256",lock_sha256],
              "cwd":str(repo),"max_wall_s":300,"cpu_reserve_s":300,"max_rss_gib":2.,
              "projected_output_bytes":134217728,"deps":[],"outputs":[unit["public_receipt_path"]],
              "sources":list(sources.values()),"env":THREAD_ENV,"amendment_sha256":lock_sha256}
        path=root/".private"/"queue_specs"/f"{uid}.json";digest=new_json(path,spec)
        queue[uid]={"path":str(path),"sha256":digest,"worker_spec_sha256":prepared["sha256"]}
    public={"schema":"pcrl-incumbent-r4-final-queue-index-v1","stage_lock_sha256":lock_sha256,
            "amendment_sha256":lock_sha256,"unit_count":len(queue),"units":queue,
            "venv_entrypoint":"repository .venv/bin/python","queue_admission":"R0/R5 required before enqueue"}
    path=root/"ATTACK_QUEUE_INDEX.json";digest=new_json(path,public,private=False)
    return {"index_path":str(path),"index_sha256":digest,"unit_count":len(queue)}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("phase",choices=("prepare","finalize"))
    p.add_argument("--config");p.add_argument("--index");p.add_argument("--stage-lock");p.add_argument("--stage-lock-sha256")
    p.add_argument("--repo");p.add_argument("--python")
    a=p.parse_args()
    if a.phase=="prepare":result=prepare(a.config)
    else:result=finalize(a.index,a.stage_lock,a.stage_lock_sha256,a.python,a.repo)
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
