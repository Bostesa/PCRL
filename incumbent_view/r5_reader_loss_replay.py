"""Reconstruct retained FIT reader losses from frozen model arrays and FIT SEX.

The numerical formulas here are independent of R2 reader/scoring modules. This
worker does no fitting. It never opens ATT/INNER task or sensitive labels.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

SALT = "IVR1|FIT-crossfold|20261010|"
CLIP = 1e-12
LAW = {"row_id", "unit", "code_r1", "code_r2", "mass", "valid",
       "b0_code_r1", "b0_code_r2"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_npz(path, keys):
    with ZipFile(path) as z:
        names = [a.filename for a in z.infolist()]
        if len(names) != len(keys) or set(names) != {k+".npy" for k in keys}:
            raise PermissionError("private NPZ member allowlist differs before load")
    with np.load(path, allow_pickle=False) as a:
        return {k:np.array(a[k],copy=True) for k in keys}


def fold_of(groups):
    return np.asarray([int(hashlib.sha256((SALT+str(v)).encode()).hexdigest(),16)%3
                       for v in groups], dtype=np.int64)


def state_id(law, bank, view):
    if view == "r1": return law["code_r1"], len(bank["recipients"]["1"]["probability"])
    if view == "r2": return law["code_r2"], len(bank["recipients"]["2"]["probability"])
    if view == "pair":
        z1=len(bank["recipients"]["1"]["probability"])
        z2=len(bank["recipients"]["2"]["probability"])
        return law["code_r1"]*z2+law["code_r2"], z1*z2
    raise PermissionError("unknown reader view")


def public_feature(bank, view, states):
    """Categorical ID, decoded probability, decision, then little-endian bits."""
    output=[]
    for j in ((1,2) if view=="pair" else (int(view[-1]),)):
        table=np.asarray(bank["recipients"][str(j)]["probability"],np.float64)
        z=len(table)
        if view=="pair":
            z2=len(bank["recipients"]["2"]["probability"])
            code=states//z2 if j==1 else states%z2
        else:code=states
        bits=(z-1).bit_length()
        decision=table[code].argmax(1)
        output.extend((np.eye(z,dtype=np.float64)[code], table[code],
                       np.eye(table.shape[1],dtype=np.float64)[decision],
                       ((code[:,None]>>np.arange(bits))&1).astype(np.float64)))
    return np.concatenate(output,axis=1)


def nonlinear(features, archive, prefix):
    mean=archive[prefix+"_mean"]
    scale=archive[prefix+"_scale"]
    w=[archive[prefix+f"_w{k}"] for k in range(6)]
    h=np.maximum(((features-mean)/scale)@w[0]+w[1],0)
    h=np.maximum(h@w[2]+w[3],0)
    logit=(h@w[4]+w[5]).ravel()
    out=np.empty(len(logit),np.float64)
    positive=logit>=0
    out[positive]=1/(1+np.exp(-logit[positive]))
    exp=np.exp(logit[~positive]);out[~positive]=exp/(1+exp)
    return out


def independent_loss_for_reader(reader_index, rec, arrays, sex, fold, view_cache):
    view=rec["view"]
    family=rec["name"].rsplit("/",1)[-1]
    ids,width,bank=view_cache[view]
    if ids.shape!=(len(sex),16) or np.any(ids<0) or np.any(ids>=width):
        raise PermissionError("actual reader state/action shape differs")
    recomputed=np.empty((len(sex),16),np.float64)
    count_summary={}
    for k in range(3):
        train=fold!=k;test=fold==k
        prior=float((int(sex[train].sum())+1)/(int(train.sum())+2))
        prefix=f"r{reader_index}_f{k}"
        if family=="prior":
            saved=float(arrays[prefix][0])
            if abs(saved-prior)>1e-14:raise ArithmeticError("FIT-only prior differs")
            probability=np.full(width,prior,np.float64)
        elif family=="categorical":
            count=np.asarray(arrays[prefix],np.float64)
            if count.shape!=(width,2) or np.any(count<0):
                raise PermissionError("saved crossfit count table differs")
            totals=count.sum(0)
            expected=np.bincount(sex[train],minlength=2).astype(np.float64)
            if np.max(np.abs(totals-expected))>1e-7:
                raise ArithmeticError("crossfit counts include wrong person/SEX mass")
            denominator=count.sum(1)
            probability=np.where(denominator>0,(count[:,1]+1)/(denominator+2),prior)
            count_summary[str(k)]={"train_rows":int(train.sum()),
                                   "count_mass_gap":float(np.max(np.abs(totals-expected)))}
        elif family=="nonlinear":
            # The frozen reader is a function of public state. Evaluate only
            # states actually reachable in this held-out fold; this is exact
            # and avoids a dense 80k-state pair feature matrix per model.
            used=np.unique(ids[test])
            features=public_feature(bank,view,used)
            predicted=nonlinear(features,arrays,prefix)
            probability=predicted[np.searchsorted(used,ids[test])]
        else:raise PermissionError("reader family differs")
        p=np.clip(probability if family=="nonlinear" else probability[ids[test]],CLIP,1-CLIP)
        recomputed[test]=np.where(sex[test,None]==1,-np.log(p),-np.log1p(-p))
    stored=np.asarray(arrays[f"r{reader_index}_loss"],np.float64)
    if stored.shape!=recomputed.shape or not np.isfinite(stored).all():
        raise PermissionError("retained reader loss shape differs")
    gap=float(np.max(np.abs(stored-recomputed)))
    if gap>1e-10:raise ArithmeticError("independent model-to-loss replay differs")
    return gap,count_summary


def run(config_path):
    active=assert_active_from_env(stage="verification")
    state=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    pinned_sources={x["path"]:x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    def pin(path):
        p=Path(path)
        if pinned_sources.get(str(p))!=sha(p):raise PermissionError("unregistered source bytes")
        return p
    cfg=json.loads(pin(config_path).read_text())
    if cfg.get("schema")!="IVR1-r5-reader-loss-replay-v1" or cfg.get("unit_id")!=active["unit_id"]:
        raise PermissionError("reader-loss replay config differs")
    unit=json.loads(pin(cfg["r2_receipt"]).read_text())
    if unit.get("source_unit_id")!=cfg["r2_unit_id"] or unit.get("status") not in (
            "COMPLETE_SELECTED","COMPLETE_IDENTITY_ALIAS","COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK"):
        raise PermissionError("selected R2 reader source incomplete")
    if not any(x["sha256"]==unit.get("source_lock_sha256") and "protection" in x["stages"]
               for x in state.get("amendments",[])):
        raise PermissionError("selected FIT science lock not registered")
    source_map=json.loads(pin(cfg["r1_source_map"]).read_text())
    sex_entry=source_map["roles"]["OSF_DEFENSE_FIT"]["sex"]
    if sex_entry["path"]!=cfg["fit_sex"] or sex_entry["sha256"]!=sha(pin(cfg["fit_sex"])):
        raise PermissionError("FIT SEX role-only export differs")
    index=json.loads(pin(cfg["private_archive_index"]).read_text())
    if (index.get("schema")!="IVR1-r2-private-checkpoint-reader-index-v1"
            or sha(cfg["private_archive_index"])!=unit["archive"]["archive_index_sha256"]
            or index.get("reader_file_sha256")!=sha(pin(cfg["readers_archive"]))
            or index["reader_file_sha256"]!=unit["archive"]["reader_file_sha256"]):
        raise PermissionError("retained reader index/file differs")
    law=read_npz(pin(cfg["fit_law"]),LAW)
    fit_release=next((x for x in unit["role_releases"]
                      if x["role"]=="OSF_DEFENSE_FIT" and x["realization"]==0),None)
    if fit_release is None or fit_release["law_file_sha256"]!=sha(cfg["fit_law"]):
        raise PermissionError("FIT action law not selected R2 source")
    labels=read_npz(pin(cfg["fit_sex"]),{"row_id","unit","sex"})
    if (len(labels["sex"])!=15434 or
            not np.array_equal(law["row_id"],labels["row_id"]) or
            not np.array_equal(law["unit"],labels["unit"]) or
            labels["sex"].dtype!=np.int64 or
            not np.isin(labels["sex"],(0,1)).all()):
        raise PermissionError("FIT sensitive role-only alignment/schema differs")
    fold=fold_of(law["unit"])
    bank=json.loads(pin(cfg["public_bank"]).read_text())
    if (sha(cfg["public_bank"])!=unit["public_bank_sha256"] or
            bank["seed"]!=unit["source_seed"]):
        raise PermissionError("frozen public bank identity differs")
    expected={f"r{i}_loss" for i in range(len(index["readers"]))}
    expected.update(key for rec in index["readers"] for key in rec["model_arrays"])
    arrays=read_npz(pin(cfg["readers_archive"]),expected)
    view_cache={}
    for view in ("r1","r2","pair"):
        ids,width=state_id(law,bank,view)
        view_cache[view]=(ids,width,bank)
    gaps={}; count_rechecks=0; max_count_gap=0.
    for i,rec in enumerate(index["readers"]):
        gap,count=independent_loss_for_reader(i,rec,arrays,labels["sex"],fold,view_cache)
        gaps[rec["name"]]=gap
        count_rechecks+=len(count)
        max_count_gap=max(max_count_gap,*(x["count_mass_gap"] for x in count.values())) if count else max_count_gap
    result={"schema":"IVR1-r5-independent-reader-model-loss-replay-v1", "status":"PASS",
            "unit_id":active["unit_id"],"r2_unit_id":cfg["r2_unit_id"],
            "reader_models":len(gaps),"fold_model_replays":3*len(gaps),
            "categorical_count_mass_rechecks":count_rechecks,
            "max_model_loss_gap":max(gaps.values()),"max_count_mass_gap":max_count_gap,
            "fit_sex_read":True,"attack_or_inner_sex_read":False,"new_reader_fits":0,
            "r2_unit_receipt_sha256":sha(cfg["r2_receipt"]),
            "reader_archive_sha256":sha(cfg["readers_archive"]),
            "limitation":"Categorical counts are checked by fold/class mass; later-round per-state count training laws are not independently refit."}
    with os.fdopen(os.open(cfg["output"],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644),"w") as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    result=run(p.parse_args().config)
    print(json.dumps({k:result[k] for k in ("status","unit_id","reader_models","max_model_loss_gap")},sort_keys=True))


if __name__=="__main__":main()
