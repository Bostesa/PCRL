"""Independent FIT fold, task, retained-reader and nomination replay.

Queue verification only. Frozen R2 artifacts are inputs; no R2/R3 scoring,
selection, reader fitting or policy code is imported here.
"""
import argparse
import hashlib
import json
from math import fsum, log
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

LAW={"row_id","unit","code_r1","code_r2","mass","valid","b0_code_r1","b0_code_r2"}
TASKS=(("income",1,2),("occupation",2,6))
METRICS=("accuracy","nll","brier")
SALT="IVR1|FIT-crossfold|20261010|"
CAP={"accuracy":.002,"nll":.001,"brier":.0005}


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def archive(path,keys):
    with ZipFile(path) as z:
        if len(z.infolist())!=len(keys) or {x.filename for x in z.infolist()}!={k+".npy" for k in keys}:
            raise PermissionError("private role archive member inventory differs")
    with np.load(path,allow_pickle=False) as f:
        return {k:np.array(f[k],copy=True) for k in keys}


def score_vector(table,code,label,metric):
    p=table[code]
    if metric=="accuracy":return (p.argmax(-1)==label[:,None]).astype(np.float64)
    if metric=="nll":
        return -np.log(np.maximum(np.take_along_axis(p,label[:,None,None],axis=2).squeeze(2),1e-12))
    return np.sum((p-np.eye(p.shape[-1])[label][:,None,:])**2,axis=2)


def paired_mean(values,mask):
    return fsum(float(x) for x in values[mask])/int(mask.sum())


def run(config_path):
    active=assert_active_from_env(stage="verification")
    state=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered={str(Path(x["path"]).resolve()):x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    def pin(path):
        p=Path(path)
        if p.is_symlink() or registered.get(str(p.resolve(strict=True)))!=sha(p):
            raise PermissionError("FIT replay source not pinned in active queue")
        return p
    cfg=json.loads(pin(config_path).read_text())
    if cfg.get("schema")!="IVR1-r5-fit-audit-v1" or cfg["unit_id"]!=active["unit_id"]:
        raise PermissionError("FIT audit unit identity differs")
    source=json.loads(pin(cfg["r2_receipt"]).read_text())
    if (source.get("schema")!="IVR1-r2-protection-unit-v1" or source["source_unit_id"]!=cfg["r2_unit_id"]
            or source["status"] not in ("COMPLETE_SELECTED","COMPLETE_IDENTITY_ALIAS","COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK")
            or source["source_seed"] not in (0,1)):
        raise PermissionError("selected R2 FIT source incomplete or different")
    science=source["source_lock_sha256"]
    if not any(x["sha256"]==science and "protection" in x["stages"] for x in state.get("amendments",[])):
        raise PermissionError("selected FIT science lock not registered")
    cross=json.loads(pin(cfg["crossfit"]).read_text())
    role_map=json.loads(pin(cfg["r1_source_map"]).read_text())
    if (role_map.get("schema")!="pcrl-incumbent-r1-private-source-paths-v1" or
            source["source_seed"] not in (0,1)):
        raise PermissionError("FIT role-only source map differs")
    for task in ("income","occupation"):
        entry=role_map["roles"]["OSF_DEFENSE_FIT"][task]
        if entry["path"]!=cfg[task] or entry["sha256"]!=sha(pin(cfg[task])):
            raise PermissionError("FIT task labels outside admitted role-only export")
    if sha(cfg["crossfit"])!=source["crossfit_manifest_sha256"] or cross["role"]!="OSF_DEFENSE_FIT":
        raise PermissionError("crossfit source differs")
    index=json.loads(pin(cfg["private_archive_index"]).read_text())
    if (index.get("schema")!="IVR1-r2-private-checkpoint-reader-index-v1"
            or sha(cfg["private_archive_index"])!=source["archive"]["archive_index_sha256"]
            or sha(pin(cfg["readers_archive"]))!=index["reader_file_sha256"]
            or index["reader_file_sha256"]!=source["archive"]["reader_file_sha256"]):
        raise PermissionError("retained FIT reader archive differs")
    law=archive(pin(cfg["fit_law"]),LAW)
    fit_release=next((x for x in source["role_releases"] if x["role"]=="OSF_DEFENSE_FIT" and x["realization"]==0),None)
    if fit_release is None or fit_release["law_file_sha256"]!=sha(cfg["fit_law"]):
        raise PermissionError("selected FIT law not in R2 UNIT receipt")
    n=len(law["row_id"])
    if n!=15434 or len(np.unique(law["unit"]))!=15428 or law["mass"].shape!=(n,16):
        raise PermissionError("FIT rows/groups/law width differs")
    folds=np.asarray([int(hashlib.sha256((SALT+str(v)).encode()).hexdigest(),16)%3 for v in law["unit"]],np.int64)
    paired=array_sha(np.c_[law["row_id"],law["unit"]]);foldsha=array_sha(np.c_[law["unit"],folds])
    if paired!=cross["row_unit_pair_sha256"] or foldsha!=cross["unit_fold_sha256"]:
        raise PermissionError("group-disjoint FIT folds differ")
    order=np.argsort(law["unit"],kind="stable")
    ordered_group=law["unit"][order];ordered_fold=folds[order]
    if np.any((ordered_group[1:]==ordered_group[:-1]) & (ordered_fold[1:]!=ordered_fold[:-1])):
        raise PermissionError("person group leaks across folds")
    bank=json.loads(pin(cfg["public_bank"]).read_text())
    if bank["seed"]!=source["source_seed"] or sha(cfg["public_bank"])!=source["public_bank_sha256"]:
        raise PermissionError("selected original public table differs")
    masks={"all":np.ones(n,bool),**{str(k):folds==k for k in range(3)}}
    result={"schema":"IVR1-r5-independent-fit-audit-v1","status":"PASS","unit_id":active["unit_id"],
            "r2_unit_id":source["source_unit_id"],"arm_id":source["arm_id"],"source_seed":source["source_seed"],
            "science_lock_sha256":science,"r2_receipt_sha256":sha(cfg["r2_receipt"]),
            "crossfit_sha256":sha(cfg["crossfit"]),"task":{},"local_readers":{},"nomination":{},
            "max_task_gap":0.,"max_reader_gap":0.,"no_attack_or_inner_labels_read":True}
    q=law["mass"]
    if (q.dtype!=np.float64 or law["valid"].dtype!=np.bool_ or
            not np.isfinite(q).all() or np.any(q<0) or
            np.any(q[~law["valid"]]!=0) or not np.allclose(q.sum(1),1,rtol=0,atol=1e-12)):
        raise PermissionError("selected FIT law invalid")
    if not np.all(law["valid"][:,0] &
            (law["code_r1"][:,0]==law["b0_code_r1"]) &
            (law["code_r2"][:,0]==law["b0_code_r2"])):
        raise PermissionError("selected FIT law first action is not exact P0")
    alias=np.array_equal(q[:,0],np.ones(n)) and np.all(q[:,1:]==0)
    if alias!=source["nomination"]["exact_p0_alias"]:
        raise PermissionError("exact P0 alias declaration differs")
    for task,r,width in TASKS:
        label=archive(pin(cfg[task]),{"row_id","unit",task})
        if not np.array_equal(label["row_id"],law["row_id"]) or not np.array_equal(label["unit"],law["unit"]):
            raise PermissionError("FIT task label rows differ")
        y=label[task]
        if y.dtype!=np.int64 or np.any(y<0) or np.any(y>=width):raise PermissionError("FIT task class differs")
        table=np.asarray(bank["recipients"][str(r)]["probability"],np.float64)
        code=law[f"code_r{r}"];base=law[f"b0_code_r{r}"]
        if (code.dtype!=np.int64 or np.any(code<0) or np.any(code>=len(table)) or
                base.dtype!=np.int64 or np.any(base<0) or np.any(base>=len(table)) or
                np.any(law["valid"] & (table[code].argmax(-1)!=table[base].argmax(-1)[:,None]))):
            raise PermissionError("FIT public support or decision differs")
        support=bank["recipients"][str(r)]["support"]
        if len(support)!=len(table):raise PermissionError("frozen action support inventory differs")
        permitted=np.array([[int(c) in support[int(src)] for c in code[i]]
                            for i,src in enumerate(base)],dtype=np.bool_)
        if np.any(law["valid"] & ~permitted):
            raise PermissionError("selected FIT law escapes frozen P0 action support")
        for metric in METRICS:
            values=score_vector(table,code,y,metric)
            sign=-1. if metric=="accuracy" else 1.
            paired_values=sign*np.sum(q*(values-values[:,0,None]),axis=1)
            for group,mask in masks.items():
                key=f"{group}/r{r}_{metric}"
                got=paired_mean(paired_values,mask)
                prior=source["fit_utility"]["clauses"][key]
                gap=abs(got-prior["margin"]);result["max_task_gap"]=max(result["max_task_gap"],gap)
                passed=got<=CAP[metric]-(0. if alias else 1e-10)
                boundary=(passed!=prior["feasible"] and abs(got-(CAP[metric]-(0. if alias else 1e-10)))<=1e-10)
                if gap>1e-10 or (passed!=prior["feasible"] and not boundary):
                    raise ArithmeticError("independent FIT task proxy differs")
                result["task"][key]={"margin":got,"source_margin":prior["margin"],"feasible":passed,
                                      "boundary":"NUMERIC_BOUNDARY_UNRESOLVED" if boundary else "AGREE"}
    reader_desc=index["readers"]
    if len({rec["name"] for rec in reader_desc})!=len(reader_desc):
        raise PermissionError("retained reader names are not unique")
    with ZipFile(pin(cfg["readers_archive"])) as z:
        listed={x.filename for x in z.infolist()}
    expected={f"r{i}_loss.npy" for i in range(len(reader_desc))}
    expected.update(k+".npy" for rec in reader_desc for k in rec["model_arrays"])
    if listed!=expected:raise PermissionError("reader loss/model archive member inventory differs")
    with np.load(cfg["readers_archive"],allow_pickle=False) as archive_file:
        pair={};prior=[]
        for i,rec in enumerate(reader_desc):
            for member,expected_sha in rec["model_arrays"].items():
                if array_sha(np.asarray(archive_file[member]))!=expected_sha:
                    raise PermissionError("retained fitted reader model/count array differs")
            loss=np.asarray(archive_file[f"r{i}_loss"],np.float64)
            if loss.shape!=(n,16) or not np.isfinite(loss).all() or array_sha(loss)!=rec["loss_sha256"]:
                raise PermissionError("frozen reader loss array differs")
            if rec["view"] in ("r1","r2"):
                for group,mask in masks.items():
                    key=f"{rec['name']}/{group}"
                    got=paired_mean(np.sum(q*(loss-loss[:,0,None]),axis=1),mask)
                    source_guard=source["fit_local_guards"]["clauses"][key]
                    gap=abs(got-source_guard["ce_difference_vs_p0"])
                    result["max_reader_gap"]=max(result["max_reader_gap"],gap)
                    passed=got>=-1e-10
                    boundary=passed!=source_guard["feasible"] and abs(got+1e-10)<=1e-10
                    if gap>1e-10 or (passed!=source_guard["feasible"] and not boundary):
                        raise ArithmeticError("independent retained local CE guard differs")
                    result["local_readers"][key]={"ce_difference_vs_p0":got,"feasible":passed,
                                                  "boundary":"NUMERIC_BOUNDARY_UNRESOLVED" if boundary else "AGREE"}
            elif rec["view"]=="pair":
                value=paired_mean(np.sum(q*loss,axis=1),masks["all"])
                pair[rec["name"]]=value
                if rec["name"].endswith("/prior"):prior.append(value)
            else:raise PermissionError("retained reader view differs")
    expected_local=4*sum(rec["view"] in ("r1","r2") for rec in reader_desc)
    if (len(result["local_readers"])!=expected_local or
            len(source["fit_local_guards"]["clauses"])!=expected_local or
            len(result["task"])!=24 or len(source["fit_utility"]["clauses"])!=24):
        raise PermissionError("FIT task or complete local-reader clause inventory differs")
    nomination=source["nomination"]
    if not pair or not prior or set(pair)!=set(nomination["pair_reader_losses"]):
        raise PermissionError("complete retained pair/prior reader slate absent")
    for name,value in pair.items():
        if abs(value-nomination["pair_reader_losses"][name])>1e-10:
            raise ArithmeticError("frozen pair reader risk differs")
    strongest=min(pair.values());baseline=min(prior)
    if (abs(strongest-nomination["pair_min_ce"])>1e-10 or
            abs(baseline-nomination["constant_prior_ce"])>1e-10 or
            abs(baseline-strongest-nomination["reader_recovery_gap"])>1e-10):
        raise ArithmeticError("independent FIT reader nomination differs")
    result["nomination"]={"pair_min_ce":strongest,"constant_prior_ce":baseline,
                           "reader_recovery_gap":baseline-strongest,"retained_pair_readers":len(pair)}
    output=Path(cfg["output"])
    with os.fdopen(os.open(output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644),"w") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    result=run(a.config)
    print(json.dumps({k:result[k] for k in ("status","unit_id","max_task_gap","max_reader_gap")}))


if __name__=="__main__":main()
