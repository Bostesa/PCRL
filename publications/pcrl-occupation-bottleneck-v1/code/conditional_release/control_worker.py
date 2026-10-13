"""Queue-only matched-control fitting and role-law serialization."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from .source import load_manifest,load_allowed_role,sha_file,write_once
from .fit import prepare_inputs,archive_result,write_npz
from .preparation import constraint_rows,expected_rows,ORIGINAL_CAPS
from .controls import (ControlCeilings,ControlPolicy,fit_independent_product,fit_sequential,
                       categorical_map,template_bounds,map_law,solve_binary_control,save_control_policy)


def run(config):
    from .scheduler import assert_active_from_env
    from .lp import direct_bayes
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text())
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("control unit/lock differs")
    kind=cfg["kind"]
    if kind not in ("LOCAL","PRODUCT","DETERMINISTIC","SEQ12","SEQ21") or cfg["support"]!="NARROW":
        raise PermissionError("control parameterization not registered")
    ceilings=ControlCeilings(seconds=cfg.get("solve_time_limit_s",600.))
    if ceilings.seconds>600.:raise PermissionError("control solve ceiling enlarged")
    manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    data=prepare_inputs(manifest,cfg["seed"])
    prep=json.loads(Path(cfg["preparation_report"]).read_text())
    if data["context"].report()!=prep["context_report"]:
        raise PermissionError("control context differs from frozen preparation")
    p=data["prepared"];space=data["space"];n=len(p["b0_code_r1"])
    contexts=data["contexts"].copy();context_count=len(data["context"].labels)
    unsupported=contexts<0
    contexts[unsupported]=context_count
    valid1=space.valid1[p["b0_code_r1"]].copy();valid2=space.valid2[p["b0_code_r2"]].copy()
    if kind!="LOCAL":
        valid1[unsupported]=False;valid2[unsupported]=False
        valid1[unsupported,0]=True;valid2[unsupported,0]=True
    sex=load_allowed_role(manifest,cfg["seed"],"OSF_DEFENSE_FIT",fields=("sex",))["sex"]["sex"]
    def constraints(law):
        return constraint_rows(law,data["costs"],data["fit"]["fold"],data["references"],data["reader_losses"],cfg["budget"])[0]
    private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    emitted1,emitted2,identity1,identity2=(p[k] for k in ("code_r1","code_r2","b0_code_r1","b0_code_r2"))
    archives=[];policy=None;selected=None
    if kind in ("LOCAL","PRODUCT"):
        if kind=="LOCAL":
            own1,groups1=np.unique(identity1,return_inverse=True)
            own2,groups2=np.unique(identity2,return_inverse=True)
        else:groups1=groups2=contexts
        result=fit_independent_product(emitted1,emitted2,identity1,identity2,groups1,groups2,valid1,valid2,
                                      sex,constraints,ceilings)
        for index,attempt in enumerate(result["attempts"]):
            name=f"COORD_{index:02d}"
            archives.append({"start":attempt["start"],"round":attempt["round"],"recipient":attempt["recipient"],
                             "solve":archive_result(private,name,attempt["result"])})
        if result["success"]:
            selected=result["selected"];factor1,factor2=selected["factors"]
            if kind=="LOCAL":
                table1=np.zeros((len(space.probability1),4));table1[:,0]=1.
                table2=np.zeros((len(space.probability2),4));table2[:,0]=1.
                table1[own1]=factor1;table2[own2]=factor2
                policy=ControlPolicy(space,kind,None,table1,table2)
            else:policy=ControlPolicy(space,kind,data["context"],factor1[:context_count],factor2[:context_count])
    elif kind=="DETERMINISTIC":
        mapping=categorical_map(contexts,16)
        valid=(valid1[:,:,None]&valid2[:,None,:]).reshape(n,16)
        law=map_law(emitted1,emitted2,identity1,identity2,mapping,template_bounds(contexts,valid),width=16)
        result=solve_binary_control(law,sex,constraints(law),ceilings)
        archives.append({"stage":"relaxation","solve":archive_result(private,"RELAXATION",result["relaxation"])})
        archives.append({"stage":"binary","solve":archive_result(private,"BINARY",result)})
        if result["unsnapped_primal"] is not None:write_npz(private/"BINARY_RAW_SOLVER_PRIMAL.npz",{"primal":result["unsnapped_primal"]})
        if result["success"]:
            objective,views=direct_bayes(law,result["theta"],sex)
            selected={"law":law,"theta":result["theta"],"objective":objective,"views":views}
            policy=ControlPolicy(space,kind,data["context"],result["theta"].reshape(-1,16)[:context_count])
    else:
        first=1 if kind=="SEQ12" else 2
        result=fit_sequential(emitted1,emitted2,identity1,identity2,contexts,valid1,valid2,sex,constraints,first,ceilings)
        archives.append({"stage":"first","solve":archive_result(private,"FIRST",result["first_result"])})
        if result["second_result"] is not None:
            archives.append({"stage":"second","solve":archive_result(private,"SECOND",result["second_result"])})
        if result["success"]:
            selected=result
            take=result["second_group_keys"][:,0]<context_count
            policy=ControlPolicy(space,"SEQUENTIAL",data["context"],result["first_theta"][:context_count],
                                 result["second_theta"][take],result["second_group_keys"][take],first)
    record={"schema":"CBR1-matched-control-unit-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],"kind":kind,
            "budget":cfg["budget"],"support":"NARROW","source_manifest_sha256":cfg["source_manifest_sha256"],
            "science_lock_sha256":cfg["science_lock_sha256"],"solver_class":result["solver_class"],
            "solves":archives,"solver_calls":len(archives),"closed_role_access":False,"inner_sex_access":False,
            "maximum_planned_solver_calls":16 if kind in ("LOCAL","PRODUCT") else 2,
            "skipped_solver_slots":(16 if kind in ("LOCAL","PRODUCT") else 2)-len(archives),
            "status":"INCOMPLETE_CONTROL" if policy is None else "COMPLETE", "global_optimality_claimed":False}
    if policy is not None:
        law,theta=selected["law"],selected["theta"]
        q=law.probabilities(theta);deployed=policy.law_prepared(p)
        gap=float(np.max(np.abs(q-deployed)))
        if gap>1e-12:raise AssertionError("control optimization/deployment law differs")
        rows=constraints(law);margins=rows.rhs-rows.matrix@theta
        if float(margins.min()) < -ceilings.feasibility_tolerance:
            raise AssertionError("executable control violates fitting constraints")
        arm_id={"LOCAL":"L_SELECTED","PRODUCT":"P_SELECTED","DETERMINISTIC":"D_SELECTED",
                "SEQ12":"S12_SELECTED","SEQ21":"S21_SELECTED"}[kind]
        package,receipt_sha=save_control_policy(private/arm_id,policy,seed=cfg["seed"],arm_id=arm_id,
                                               budget=cfg["budget"],source_manifest=manifest)
        role_laws=[]
        for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
            source=data["fit"] if role=="OSF_DEFENSE_FIT" else load_allowed_role(manifest,cfg["seed"],role)
            wire=source["wire"];e1,e2,mass=policy.law_codes(wire["code_r1"],wire["code_r2"])
            role_sha=write_npz(private/arm_id/(role+"_LAW.npz"),{"row_id":wire["row_id"],"unit":wire["unit"],
                                                                       "code_r1":e1,"code_r2":e2,"mass":mass})
            role_laws.append({"role":role,"law_sha256":role_sha,"rows":len(mass)})
        inner=load_allowed_role(manifest,cfg["seed"],"INNER_SELECTION",fields=("income","occupation","references"))
        ip=space.prepare(inner["wire"]["code_r1"],inner["wire"]["code_r2"]);iq=policy.law_prepared(ip)
        utility=expected_rows(ip,iq,space,(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
        gates=[];private_utility={}
        for ref,values in utility["excess"].items():
            for key,value in values.items():
                metric=key.rsplit("_",1)[1];margin=float(value.mean());cap=ORIGINAL_CAPS[metric]
                gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
                private_utility[f"{ref}_{key}"]=value
        write_npz(private/arm_id/"INNER_UTILITY_ROWS.npz",{"row_id":inner["wire"]["row_id"],"unit":inner["wire"]["unit"],**private_utility})
        change=float(law.change_coefficients()[0]+law.change_coefficients()[1]@theta)
        decisions=(space.probability1[identity1].argmax(1),space.probability2[identity2].argmax(1))
        decision_views={}
        for name,key in (("r1",decisions[0][:,None]),("r2",decisions[1][:,None]),("pair",np.c_[*decisions])):
            unique,inverse=np.unique(key,axis=0,return_inverse=True)
            count=np.stack([np.bincount(inverse,weights=(sex==b)/n,minlength=len(unique)) for b in (0,1)])
            decision_views[name]=float(count.max(0).sum())
        lower_bound=decision_views["pair"]+.5*(decision_views["r1"]+decision_views["r2"])
        record.update(arm_id=arm_id,fitted_objective=selected["objective"],fitted_recovery=selected["views"],
                      actual_change_rate=change,minimum_fitting_constraint_margin=float(margins.min()),
                      optimization_deployment_max_gap=gap,policy_receipt_sha256=receipt_sha,
                      policy_package_sha256=package["package_sha256"],parameter_count=package["parameter_count"],
                      role_laws=role_laws,inner_point_utility=gates,inner_point_eligible=all(x["pass"] for x in gates))
        record.update(decision_only_relaxation_bound=lower_bound,
                      decision_only_relaxation_scope="remove utility/CE caps and retain only mandatory same-class decisions",
                      heuristic_global_gap_upper_bound=max(0.,selected["objective"]-lower_bound),
                      exact_remaining_global_gap=None)
    public=Path(cfg["public_output"]);public.mkdir(parents=True,exist_ok=True)
    write_once(public/(cfg["unit_id"]+"_RECEIPT.json"),record,False)
    return {"unit_id":cfg["unit_id"],"status":record["status"],"solver_calls":record["solver_calls"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();print(json.dumps(run(args.config)))
