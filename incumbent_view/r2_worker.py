"""One guarded R2 FIT arm and immutable role-only release materialization.

The queue pins all code/config/source bytes before ANY role label or code read.
No assessment or HEAD_VALIDATION path is accepted. ATT/INNER read only P0
codes/probabilities/decisions, never task or sensitive labels.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r1_bank import load_bank,sha_file
from .r2_package import array_sha,from_bank,read_exact,save_policy,write_json,write_npz
from .r2_space import group_folds,decimal_cost_check,costs,accurate_margin,identity_q,utility_receipt
from .r2_train import fit_arm
from .r2_materialize import materialize
from .r2_readers import reader_guard_receipt,nomination

P0_KEYS={"row_id","unit","code_r1","code_r2","prob_r1","prob_r2","decision_r1","decision_r2"}
FIT_ROWS=15434
ROLES=("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION")
ARM_IDS=("O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")


def _pinned(record,registered):
    path=Path(record["path"])
    if path.is_symlink():raise PermissionError("symlinked scientific source refused")
    path=path.resolve(strict=True)
    if registered.get(path)!=record["sha256"] or sha_file(path)!=record["sha256"]:
        raise PermissionError("active-queue source byte pin differs")
    return path


def _labels(record,registered,name,row,unit):
    path=_pinned(record,registered)
    item=read_exact(path,{"row_id","unit",name})
    if not np.array_equal(item["row_id"],row) or not np.array_equal(item["unit"],unit):
        raise PermissionError("authorized FIT labels/people differ")
    value=item[name]
    if value.dtype!=np.int64 or value.shape!=row.shape:raise ValueError("FIT label class/order differs")
    return value


def _policy_archive(directory,summary):
    ck=summary.pop("checkpoints",[]);readers=summary.pop("readers",[])
    arrays={};catalog=[]
    for i,item in enumerate(ck):
        saved={k:v for k,v in item.items() if k not in ("weights","policy","final_policy")}
        for j,w in enumerate(item["weights"]):
            for k,a in enumerate(w):arrays[f"c{i}_w{j}_{k}"]=a
        catalog.append(saved)
    check_sha=write_npz(directory/"CHECKPOINTS.npz",arrays) if arrays else None
    # Every reader loss and all weighted model counts/weights remain private.
    reader_arrays={};reader_catalog=[]
    for i,r in enumerate(readers):
        reader_arrays[f"r{i}_loss"]=r.loss
        descriptor={"name":r.name,"view":r.view,"round":r.round_index,"start":r.start_index,
                    "baseline":r.baseline,"loss_sha256":array_sha(r.loss),"model_arrays":{}}
        for fold,model in r.models.items():
            if isinstance(model,dict):
                for key,value in model.items():
                    name=f"r{i}_{fold}_{key}";reader_arrays[name]=value
                    descriptor["model_arrays"][name]=array_sha(value)
            else:
                name=f"r{i}_{fold}";reader_arrays[name]=model
                descriptor["model_arrays"][name]=array_sha(model)
        reader_catalog.append(descriptor)
    reader_sha=write_npz(directory/"READERS.npz",reader_arrays) if reader_arrays else None
    index={"schema":"IVR1-r2-private-checkpoint-reader-index-v1",
           "checkpoint_file_sha256":check_sha,"reader_file_sha256":reader_sha,
           "checkpoints":catalog,"readers":reader_catalog}
    private_index_sha=write_json(directory/"PRIVATE_ARCHIVE_INDEX.json",index)
    return {"checkpoint_file_sha256":check_sha,"reader_file_sha256":reader_sha,
            "archive_index_sha256":private_index_sha,"checkpoint_count":len(ck),
            "reader_count":len(readers),"reader_model_fold_count":sum(len(r.models) for r in readers)}


def _release_inputs(cfg,registered):
    out={}
    for role in ROLES:
        item=cfg["role_p0"][role]
        p=_pinned(item["release"],registered)
        rec=json.loads(_pinned(item["receipt"],registered).read_text())
        if rec.get("release_sha256")!=sha_file(p):raise PermissionError("R1 role release receipt differs")
        out[role]=read_exact(p,P0_KEYS)
    return out


def _same_wire_source(space,source):
    n=len(source["row_id"])
    if any(source[k].dtype!=np.int64 or source[k].shape!=(n,) for k in ("row_id","unit","code_r1","code_r2","decision_r1","decision_r2")):
        raise PermissionError("P0 role wire integer schema differs")
    if len(np.unique(source["row_id"]))!=n:raise PermissionError("P0 role rows duplicated")
    for r in (1,2):
        code=source[f"code_r{r}"];table=getattr(space,f"probability{r}")
        if np.any(code<0) or np.any(code>=len(table)) or not np.array_equal(source[f"prob_r{r}"],table[code]) or not np.array_equal(source[f"decision_r{r}"],table[code].argmax(1)):
            raise PermissionError("P0 original-ID decoder wire differs")


def run(config_path):
    active=assert_active_from_env(stage="protection")
    queue=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered={Path(item["path"]).resolve():item["sha256"] for item in queue["units"][active["unit_id"]]["spec"]["sources"]}
    config_path=Path(config_path).resolve(strict=True)
    if registered.get(config_path)!=sha_file(config_path):raise PermissionError("unit config unpinned")
    cfg=json.loads(config_path.read_text())
    arm,seed=cfg["arm_id"],cfg["source_seed"]
    if (cfg.get("schema")!="IVR1-r2-fit-config-v1" or cfg["unit_id"]!=active["unit_id"]
        or arm not in ARM_IDS or seed not in (0,1) or cfg["source_lock_sha256"]!=active["seal_sha256"]):
        raise PermissionError("R2 scientific unit configuration differs")
    pins={name:_pinned(rec,registered) for name,rec in cfg["pins"].items()}
    if (sha_file(pins["campaign_lock"])!=active["seal_sha256"] or
        sha_file(pins["stream_roots"])!=cfg["stream_roots_sha256"] or
        sha_file(pins["stream_manifest"])!=cfg["stream_manifest_sha256"]):
        raise PermissionError("protection lock/streams byte binding differs")
    bank=load_bank(pins["public_bank"],cfg["pins"]["public_bank"]["sha256"])
    if bank["seed"]!=seed or bank["source_binding_sha256"]!=cfg["source_binding_sha256"]:
        raise PermissionError("original P0 bank/seed binding differs")
    roots=json.loads(pins["stream_roots"].read_text())
    stream=json.loads(pins["stream_manifest"].read_text())
    if (stream["copied_private_file_sha256"]!=sha_file(pins["stream_roots"]) or
        stream["source_private_file_sha256"]!=sha_file(pins["stream_roots"])):
        raise PermissionError("private root copy differs from source")
    release_roots={i:bytes.fromhex(roots["release_realization_roots_hex"][str(i)]) for i in range(3)}
    if any(hashlib.sha256(v).hexdigest()!=stream["realization_root_sha256"][str(i)] for i,v in release_roots.items()):
        raise PermissionError("private HMAC root commitment differs")
    # All source/provenance byte checks precede any person-array read.
    source=_release_inputs(cfg,registered)
    space=from_bank(bank)
    for role,value in source.items():_same_wire_source(space,value)
    fit=source["OSF_DEFENSE_FIT"]
    if len(fit["row_id"])!=FIT_ROWS or np.unique(fit["unit"]).size!=15428:
        raise PermissionError("fixed FIT people/groups differ")
    y1=_labels(cfg["fit_labels"]["income"],registered,"income",fit["row_id"],fit["unit"])
    y2=_labels(cfg["fit_labels"]["occupation"],registered,"occupation",fit["row_id"],fit["unit"])
    sex=_labels(cfg["fit_labels"]["sex"],registered,"sex",fit["row_id"],fit["unit"])
    if not np.isin(sex,(0,1)).all():raise PermissionError("protected fitting class differs")
    fold=group_folds(fit["unit"])
    if len(np.unique(fold))!=3:raise PermissionError("FIT crossfit folds incomplete")
    crossfit={"schema":"IVR1-r2-fit-crossfold-v1","salt":"IVR1|FIT-crossfold|20261010|",
              "source_seed":seed,"row_unit_pair_sha256":array_sha(np.c_[fit["row_id"],fit["unit"]]),
              "unit_fold_sha256":array_sha(np.c_[fit["unit"],fold]),
              "fold_rows":{str(k):int(np.sum(fold==k)) for k in range(3)},
              "fold_groups":{str(k):int(np.unique(fit["unit"][fold==k]).size) for k in range(3)},
              "role":"OSF_DEFENSE_FIT","attack_fit_or_inner_labels_read":False}
    public_root=Path(cfg["public_output"]);public_root.mkdir(parents=True,exist_ok=True)
    crossfit_path=public_root/f"{cfg['unit_id']}_CROSSFIT.json"
    crossfit_sha=write_json(crossfit_path,crossfit,private=False)
    private=Path(cfg["private_output"])
    if private.exists() or ".private" not in private.parts:raise PermissionError("new owner-only unit directory required")
    private.mkdir(parents=True,mode=0o700)
    exact=space.prepare(fit["code_r1"],fit["code_r2"])
    active_pairs=np.unique(np.c_[fit["code_r1"],fit["code_r2"]],axis=0).shape[0]
    support_widths=np.sum(exact["valid"],axis=1)
    started=time.process_time()
    policy,summary=fit_arm(space,arm,fit["code_r1"],fit["code_r2"],y1,y2,sex,fit["unit"],
                            seed=cfg["recipe_seed"],cpu_seconds=cfg["fit_cpu_ceiling_s"])
    if summary["status"]=="COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK":
        # A frozen sequential first stage cannot be retroactively edited. The
        # separately archived whole-P0 policy remains an honest alias with
        # complete task/reader receipts and a named dependency failure.
        q0=identity_q(exact);reader_slate=summary["readers"]
        summary["selection"]={"start":"separate_whole_p0_fallback","update":0,
                              "dependency_failure_stage":summary["stage"]}
        summary["utility"]=utility_receipt(q0,costs(space,exact,y1,y2),fold)
        summary["guards"]=reader_guard_receipt(q0,reader_slate,fold)
        summary["nomination"]=nomination(q0,reader_slate)
    archive=_policy_archive(private,summary)
    status=summary["status"]
    public={"schema":"IVR1-r2-protection-unit-v1","source_unit_id":cfg["unit_id"],"arm_id":arm,
            "source_seed":seed,"status":status,"source_lock_sha256":active["seal_sha256"],
            "source_binding_sha256":bank["source_binding_sha256"],"public_bank_sha256":sha_file(pins["public_bank"]),
            "stream_manifest_sha256":sha_file(pins["stream_manifest"]),"crossfit_manifest_sha256":crossfit_sha,
            "crossfit_manifest_relative":f"r2/{crossfit_path.name}","fit_rows":len(fit["row_id"]),
            "fit_groups":len(np.unique(fit["unit"])),"reachable_p0_pair_count":int(active_pairs),
            "valid_action_width_counts":{str(k):int(np.sum(support_widths==k)) for k in np.unique(support_widths)},
            "archive":archive,"rounds":summary["rounds"],"optimization_cpu_seconds":summary["elapsed_cpu_seconds"],
            "whole_unit_cpu_seconds_before_receipt":time.process_time()-started}
    if policy is not None:
        q=policy.law_prepared(exact)
        decimal=decimal_cost_check(space,exact,q,y1,y2)
        float64={k:accurate_margin(q,v,k)[0] for k,v in costs(space,exact,y1,y2).items()}
        numeric_error={k:abs(decimal[k]-float64[k]) for k in decimal}
        if max(numeric_error.values())>1e-10:raise RuntimeError("selected FIT cost exceeds prelocked numeric envelope")
        alias=bool(np.array_equal(q,identity_q(exact)))
        nomination=summary["nomination"]
        summary_selection=summary["selection"]
        p_receipt,p_sha=save_policy(private/"policy",policy,arm_id=arm,source_seed=seed,
                                   source_unit_id=cfg["unit_id"],bank_sha256=sha_file(pins["public_bank"]),
                                   source_binding_sha256=bank["source_binding_sha256"],lock_sha256=active["seal_sha256"],
                                   selection=summary_selection,crossfit_manifest_sha256=crossfit_sha,exact_p0_alias=alias)
        release_records=[]
        for role,role_source in source.items():
            for draw in range(3):
                name=f"{role}_R{draw}"
                r_receipt,r_sha=materialize(private/name,policy,role_source,bank,arm_id=arm,
                              source_seed=seed,source_unit_id=cfg["unit_id"],role=role,realization=draw,
                              private_root=release_roots[draw],policy_receipt_sha256=p_sha)
                release_records.append({"role":role,"realization":draw,"private_relative_dir":f"r2/.private/units/{cfg['unit_id']}/{name}",
                                        "receipt_sha256":r_sha,"law_file_sha256":r_receipt["law_file_sha256"],
                                        "codes_file_sha256":r_receipt["codes_file_sha256"]})
        public.update({"selected_policy_receipt_sha256":p_sha,"selected_policy_package_sha256":p_receipt["policy_package_sha256"],
                       "selected_private_relative_dir":f"r2/.private/units/{cfg['unit_id']}/policy",
                       "selection":summary_selection,"nomination":{**nomination,"exact_p0_alias":alias,
                             "crossfit_manifest_sha256":crossfit_sha},"fit_utility":summary["utility"],
                       "fit_local_guards":summary["guards"],"decimal_float64_error":numeric_error,
                       "role_releases":release_records})
    else:
        public.update({"selected_policy_receipt_sha256":None,"selection":None,"role_releases":[],
                       "incomplete":True,"checkpoint_preserved":archive["checkpoint_count"]})
    path=public_root/f"{cfg['unit_id']}_UNIT_RECEIPT.json"
    receipt_sha=write_json(path,public,private=False)
    print(json.dumps({"status":status,"unit_id":cfg["unit_id"],"unit_receipt_sha256":receipt_sha,
                      "selected_policy_receipt_sha256":public["selected_policy_receipt_sha256"]}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();run(args.config)


if __name__=="__main__":main()
