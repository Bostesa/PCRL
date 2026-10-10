"""Queued, label-free exact P0 raw source admission and old-space diagnosis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np

from dpc.partition import assign_fine
from qpc import deploy as QD
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r1_bank import nearest_support, sha_array, sha_file, wire_hash, wires_for_codes
from .r1_deploy import P0Service


FEATURES = {"row_id", "unit", "X", "feature_names"}
REFERENCE = {"row_id", "unit", "prob_r1", "prob_r2", "decision_r1", "decision_r2", "code_r1", "code_r2"}


def read_exact(path, keys):
    with ZipFile(path) as z:
        if len(z.infolist()) != len(keys) or {a.filename for a in z.infolist()} != {k+".npy" for k in keys}:
            raise ValueError("role-only archive has unexpected fields")
    with np.load(path, allow_pickle=False) as a:
        return {k: np.array(a[k], copy=True) for k in keys}


def write_json(path, value, mode=0o644):
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, mode)
    with os.fdopen(fd,"w") as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write("\n")


def write_npz(path, **values):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as out:
        np.savez_compressed(out,**values)


def collisions(input_rows, output_codes):
    buckets={}
    for row,code in zip(input_rows,output_codes):
        buckets.setdefault(np.ascontiguousarray(row).tobytes(),set()).add(tuple(np.atleast_1d(code).tolist()))
    conflicting=[v for v in buckets.values() if len(v)>1]
    return {"rows":len(input_rows),"unique_inputs":len(buckets),"repeated_rows":len(input_rows)-len(buckets),
            "conflicting_inputs":len(conflicting),"required_output_states_in_conflicts":sum(len(v) for v in conflicting)}


def run(config_path):
    active=assert_active_from_env(stage="admission")
    state=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered={Path(v["path"]).resolve():v["sha256"] for v in state["units"][active["unit_id"]]["spec"]["sources"]}
    def pinned(path):
        p=Path(path).resolve(strict=True)
        if registered.get(p)!=sha_file(p):raise PermissionError("source is not active-queue byte-pinned")
        return p
    cfg=json.loads(pinned(config_path).read_text())
    if cfg["unit_id"]!=active["unit_id"] or cfg["schema"]!="pcrl-incumbent-r1-admission-config-v1":
        raise PermissionError("source admission config identity differs")
    for item in cfg["pins"].values():
        if sha_file(pinned(item["path"]))!=item["sha256"]:raise ValueError("source hash differs")
    decoder=json.loads(pinned(cfg["pins"]["decoder"]["path"]).read_text())
    bank={"schema":"pcrl-incumbent-original-public-bank-v1","seed":cfg["seed"],
          "release_id":cfg["release_id"],"source_binding_sha256":cfg["source_binding_sha256"],
          "feature_names":cfg["feature_names"],"feature_schema_sha256":cfg["feature_schema_sha256"],
          "decoder_embedded_sha256":decoder["decoder_sha256"],"source_pins":{k:v["sha256"] for k,v in cfg["pins"].items()},
          "support_rule":"new instantiation: identity plus <=3 nearest same-class original global IDs; symmetric KL floor/renorm 1e-12, distance/global-ID tie", "recipients":{}}
    for recipient in (1,2):
        p=np.asarray(decoder[f"r{recipient}"]["q"],dtype=np.float64)
        if sha_array(p)!=cfg["bank_sha256_by_recipient"][f"r{recipient}"]:raise ValueError("exact original decoder array differs")
        bank["recipients"][str(recipient)]={"probability":p.tolist(),"decision":p.argmax(1).tolist(),
           "global_ids":list(range(len(p))),"code_length_bits":(len(p)-1).bit_length(),"bank_sha256":sha_array(p),
           "support":[list(v) for v in nearest_support(p)]}
    service=P0Service(cfg["pins"],bank)
    old_pair=QD.load_pair(pinned(cfg["old_policy"]["path"]))
    old_archive=pinned(cfg["old_package"]["path"])
    with ZipFile(old_archive) as z:
        if any(x.filename.startswith(("sex","label","income","occupation")) for x in z.infolist()):raise ValueError("old immutable policy contains labels")
    with np.load(old_archive,allow_pickle=False) as a:
        old_banks=[np.array(a[f"bank{i}_probability"],copy=True) for i in (1,2)]
        old_actions=[np.array(a[f"actions{i}"],copy=True) for i in (1,2)]
    newroot=Path(cfg["private_output_dir"])
    if newroot.exists() or ".private" not in newroot.parts:raise PermissionError("new private output required")
    newroot.mkdir(mode=0o700)
    public={"schema":"pcrl-incumbent-r1-source-admission-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],
            "status":"PASS","label_fields_read":[],"source_pins":{k:v["sha256"] for k,v in cfg["pins"].items()},
            "states":[len(service.tables[0]),len(service.tables[1])],"roles":{},"reachability":{}}
    state_match=[]
    for p,old in zip(service.tables,old_banks):
        lookup={row.tobytes():i for i,row in enumerate(old)}
        match=np.asarray([lookup.get(row.tobytes(),-1) for row in p],dtype=np.int64);state_match.append(match)
    public["reachability"]["old_global_bank_missing_states"]=[int(np.sum(x<0)) for x in state_match]
    public["reachability"]["complete_wire_identity"]="Old W_TSTAR code/metadata contract differs; probability-state support comparison reported separately"
    for role,item in cfg["roles"].items():
        assert_active_from_env(unit_id=cfg["unit_id"],stage="admission")
        x=read_exact(pinned(item["features"]["path"]),FEATURES)
        n=len(x["row_id"])
        if n!=item["rows"] or len(np.unique(x["unit"]))!=item["groups"] or tuple(x["feature_names"].tolist())!=tuple(cfg["feature_names"]):raise ValueError("role people/schema differs")
        if sha_array(np.c_[x["row_id"],x["unit"]])!=item["row_unit_pair_sha256"]:raise ValueError("role ordered people differs")
        out,scores=service.route_raw(x["X"],x["feature_names"])
        wire_hashes={f"r{i}":wire_hash(wires_for_codes(bank,i,out[i-1][0])) for i in (1,2)}
        parity="NO_HISTORICAL_FIT_RELEASE; exact frozen route and identity checked"
        if "historical" in item:
            hist=read_exact(pinned(item["historical"]["path"]),REFERENCE)
            if not np.array_equal(hist["row_id"],x["row_id"]) or not np.array_equal(hist["unit"],x["unit"]):raise ValueError("historical people differ")
            rec=json.loads(pinned(item["historical_receipt"]["path"]).read_text())
            for i in (1,2):
                for key,value in zip(("code","prob","decision"),out[i-1]):
                    if not np.array_equal(hist[f"{key}_r{i}"],value):raise ValueError("raw P0 differs from actual historical release")
                if wire_hashes[f"r{i}"]!=rec["wire_sha256_by_recipient"][f"r{i}"]:raise ValueError("actual metadata-bearing wire hash differs")
            parity="ALL_ROWS_EXACT_HISTORICAL_CODE_PROB_DECISION_AND_METADATA_WIRE"
        role_root=newroot/role;role_root.mkdir(mode=0o700)
        arrays={"row_id":x["row_id"],"unit":x["unit"]}
        for i in (1,2):
            for key,value in zip(("code","prob","decision"),out[i-1]):arrays[f"{key}_r{i}"]=value
        path=role_root/"P0_RELEASE.npz";write_npz(path,**arrays)
        rolesummary={"rows":n,"groups":item["groups"],"row_unit_pair_sha256":item["row_unit_pair_sha256"],
          "source_features_sha256":item["features"]["sha256"],"release_sha256":sha_file(path),"wire_sha256_by_recipient":wire_hashes,
          "parity":parity,"private_relative_label":f"r1/.private/{cfg['unit_id']}/{role}/P0_RELEASE.npz"}
        write_json(role_root/"P0_ROLE_RECEIPT.json",rolesummary,0o600);public["roles"][role]=rolesummary
        joined=np.c_[scores[0],scores[1]];paircode=np.c_[out[0][0],out[1][0]]
        diagnosis={"central_old_score_inputs":collisions(joined,paircode),"local_old_score_inputs":[collisions(scores[i],out[i][0]) for i in (0,1)],"old_per_row_support_missing":[]}
        for i in (0,1):
            pol=(old_pair.p1,old_pair.p2)[i]
            cells=assign_fine(scores[i],scores[i].argmax(1),pol.fine)
            required=state_match[i][out[i][0]]
            present=np.asarray([wanted>=0 and wanted in old_actions[i][cell] for wanted,cell in zip(required,cells)])
            diagnosis["old_per_row_support_missing"].append(int(np.sum(~present)))
        public["reachability"][role]=diagnosis
        if role=="OSF_DEFENSE_FIT":
            indices=np.linspace(0,n-1,32,dtype=np.int64);sample=role_root/"sample_input.npz"
            write_npz(sample,X=x["X"][indices],feature_names=x["feature_names"])
            schema=role_root/"schema.json";write_json(schema,list(cfg["feature_names"]),0o600)
            historical=role_root/"historical_hcal_sample.npz"
            command=[cfg["python"],"-m","hcal.deploy","--unit",str(Path(cfg["pins"]["model"]["path"]).parent),"--policy",cfg["pins"]["policy"]["path"],"--decoder",cfg["pins"]["decoder"]["path"],"--decoder-sha256",decoder["decoder_sha256"],"--X",str(sample),"--schema",str(schema),"--schema-sha256",cfg["feature_schema_sha256"],"--out",str(historical),"--seed",str(cfg["seed"])]
            run=subprocess.run(command,cwd=cfg["cwd"],capture_output=True,text=True)
            if run.returncode:
                write_json(role_root/"NESTED_FAILURE.json",{"returncode":run.returncode,"stderr":run.stderr},0o600)
                raise RuntimeError("historical raw CLI failed; private diagnostic retained")
            actual=read_exact(historical,{"tokens_1","probs_1","decision_1","tokens_2","probs_2","decision_2"})
            for i in (1,2):
                for key,value in zip(("tokens","probs","decision"),out[i-1]):
                    if not np.array_equal(actual[f"{key}_{i}"],value[indices]):raise ValueError("historical CLI and new raw P0 router differ")
                wire=wires_for_codes(bank,i,out[i-1][0][indices]);target=role_root/f"recipient{i}.jsonl"
                with os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"w") as stream:
                    for row in wire:stream.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
            public["raw32_historical_hcal_cli_parity"]="PASS_BOTH_RECIPIENTS"
    write_json(cfg["public_bank"],bank)
    public["public_bank_sha256"]=sha_file(cfg["public_bank"])
    write_json(cfg["public_receipt"],public)
    return public


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument("--config",required=True)
    args=parser.parse_args();result=run(args.config)
    print(json.dumps({"unit_id":result["unit_id"],"status":result["status"],"public_bank_sha256":result["public_bank_sha256"]},sort_keys=True))


if __name__=="__main__":main()
