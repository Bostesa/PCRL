"""Queued registered WIDE matched controls; no unbounded dense state kernel."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from .source import load_manifest,load_allowed_role,sha_file,write_once
from .fit import prepare_inputs,archive_result,write_npz
from .preparation import constraint_rows,ORIGINAL_CAPS
from .templates import build_template_law
from . import memory_lp
from .reader_extension import frozen_local_loss_tables,losses_for_actions
from .controls import ControlCeilings,solve_binary_control
from .wide_control_fit import fit_product,fit_sequential,pad_template
from .control_wide_policy import ControlWidePolicy,save_wide_control_policy


def run(config, *, numerical_backend=None):
    backend=memory_lp if numerical_backend is None else numerical_backend
    from .scheduler import assert_active_from_env
    direct_bayes=backend.direct_bayes
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text())
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("wide control unit/lock differs")
    kind=cfg["kind"]
    if kind not in ("LOCAL","PRODUCT","DETERMINISTIC","SEQ12","SEQ21") or cfg["support"]!="WIDE":
        raise PermissionError("wide control family unregistered")
    ceilings=ControlCeilings(seconds=cfg.get("solve_time_limit_s",600.))
    if ceilings.seconds>600.:raise PermissionError("wide control ceiling enlarged")
    manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    data=prepare_inputs(manifest,cfg["seed"])
    prep=json.loads(Path(cfg["preparation_report"]).read_text())
    if data["context"].report()!=prep["context_report"]:
        raise PermissionError("wide control context differs")
    space=data["space"];wire=data["fit"]["wire"];c1=wire["code_r1"];c2=wire["code_r2"]
    layout_contexts=data["contexts"].copy()
    if kind=="LOCAL":
        pair_class=6*space.probability1[c1].argmax(1)+space.probability2[c2].argmax(1)
        _,layout_contexts=np.unique(pair_class,return_inverse=True)
        layout_contexts=layout_contexts.astype(np.int64)
    template,metadata=build_template_law(space,c1,c2,layout_contexts,"WIDE",cfg["maximum_input_bytes"])
    full_widths=tuple(1+np.bincount(getattr(space,f"probability{r}").argmax(1)).max() for r in (1,2))
    if kind=="LOCAL":
        # LOCAL applies to every admissible new pairing of its own-bank codes;
        # there is no central-context fallback on an unseen combination.
        global_width=int(full_widths[0]*full_widths[1])
        projected_bytes=len(c1)*global_width*84+template.parameters*64+len(c1)*64
        if projected_bytes>cfg["maximum_input_bytes"]:
            raise MemoryError("local full-bank padded input exceeds registered input ceiling")
        template,metadata=pad_template(template,metadata,global_width)
    private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    sex=load_allowed_role(manifest,cfg["seed"],"OSF_DEFENSE_FIT",fields=("sex",))["sex"]["sex"]
    tables,catalog,reader_arrays,index=frozen_local_loss_tables(manifest,cfg["seed"],space,data["fit"]["fold"],sex)
    del reader_arrays
    p={"code_r1":template.emitted1,"code_r2":template.emitted2,"b0_code_r1":c1,"b0_code_r2":c2}
    costs=backend.LazyTaskCosts(space,p,data["labels"])
    losses=losses_for_actions(tables,template.emitted1,template.emitted2,data["fit"]["fold"],sex)
    def constraints(law):
        return constraint_rows(law,costs,data["fit"]["fold"],data["references"],losses,cfg["budget"])[0]
    archives=[];selected=None;policy=None
    if kind in ("LOCAL","PRODUCT"):
        if kind=="LOCAL":
            own1,groups1=np.unique(c1,return_inverse=True);own2,groups2=np.unique(c2,return_inverse=True)
        else:
            groups1=groups2=layout_contexts.copy();groups1[groups1<0]=0
        def archive_coordinate(i,attempt):
            archives.append({"start":attempt["start"],"round":attempt["round"],"recipient":attempt["recipient"],
                             "solve":archive_result(private,f"COORD_{i:02d}",attempt["result"])})
        result=fit_product(template,metadata,layout_contexts,groups1,groups2,sex,constraints,ceilings,
                           solver_module=backend,attempt_callback=archive_coordinate)
        if result["success"]:
            selected=result["selected"];one,two=selected["factors"]
            if kind=="LOCAL":
                table1=np.zeros((len(space.probability1),int(full_widths[0])));table1[:,0]=1.
                table2=np.zeros((len(space.probability2),int(full_widths[1])));table2[:,0]=1.
                table1[own1,:one.shape[1]]=one;table2[own2,:two.shape[1]]=two
                policy=ControlWidePolicy(space,kind,None,table1,table2)
            else:
                policy=ControlWidePolicy(space,kind,data["context"],one,two,metadata=metadata)
    elif kind=="DETERMINISTIC":
        result=solve_binary_control(template,sex,constraints(template),ceilings,protection_program_builder=backend.protection_program)
        archives.append({"stage":"relaxation","solve":archive_result(private,"RELAXATION",result["relaxation"])})
        archives.append({"stage":"binary","solve":archive_result(private,"BINARY",result)})
        if result["unsnapped_primal"] is not None:write_npz(private/"BINARY_RAW_SOLVER_PRIMAL.npz",{"primal":result["unsnapped_primal"]})
        if result["success"]:
            objective,views=direct_bayes(template,result["theta"],sex)
            selected={"law":template,"theta":result["theta"],"objective":objective,"views":views}
            policy=ControlWidePolicy(space,kind,data["context"],result["theta"],metadata=metadata)
    else:
        first=1 if kind=="SEQ12" else 2
        result=fit_sequential(template,metadata,layout_contexts,sex,constraints,first,ceilings,solver_module=backend)
        archives.append({"stage":"first","solve":archive_result(private,"FIRST",result["first_result"])})
        if result["second_result"] is not None:
            archives.append({"stage":"second","solve":archive_result(private,"SECOND",result["second_result"])})
        if result["success"]:
            selected=result
            policy=ControlWidePolicy(space,"SEQUENTIAL",data["context"],result["first_theta"],result["second_theta"],
                                     result["second_group_keys"],first,metadata)
    record={"schema":"CBR1-matched-control-unit-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],"kind":kind,
            "budget":cfg["budget"],"support":"WIDE","source_manifest_sha256":cfg["source_manifest_sha256"],
            "science_lock_sha256":cfg["science_lock_sha256"],"solver_class":result["solver_class"],
            "solves":archives,"solver_calls":len(archives),"closed_role_access":False,"inner_sex_access":False,
            "maximum_planned_solver_calls":16 if kind in ("LOCAL","PRODUCT") else 2,
            "skipped_solver_slots":(16 if kind in ("LOCAL","PRODUCT") else 2)-len(archives),
            "support_metadata":metadata,"frozen_guard_catalog":catalog,
            "engineering_backend":"exact encoded-pair memory_lp with lazy task costs; mature HiGHS unchanged",
            "lp_solver_feasibility_request":1e-8 if kind=="DETERMINISTIC" or numerical_backend is None else 1e-10,
            "mip_solver_feasibility_request":1e-9,
            "status":"INCOMPLETE_CONTROL" if policy is None else "COMPLETE","global_optimality_claimed":False}
    if policy is not None:
        law,theta=selected["law"],selected["theta"]
        q=law.probabilities(theta);a,b,deployed=policy.law_codes(c1,c2)
        if not np.array_equal(a,law.emitted1) or not np.array_equal(b,law.emitted2):
            raise AssertionError("wide control optimization/deployment emitted bank differs")
        gap=float(np.max(np.abs(q-deployed)))
        if gap>1e-12:raise AssertionError("wide control optimization/deployment law differs")
        rows=constraints(law);margins=rows.rhs-rows.matrix@theta
        if float(margins.min()) < -ceilings.feasibility_tolerance:
            raise AssertionError("wide executable control violates fitting caps")
        arm_id={"LOCAL":"L_SELECTED","PRODUCT":"P_SELECTED","DETERMINISTIC":"D_SELECTED",
                "SEQ12":"S12_SELECTED","SEQ21":"S21_SELECTED"}[kind]
        package,receipt_sha=save_wide_control_policy(private/arm_id,policy,seed=cfg["seed"],arm_id=arm_id,
                                                    budget=cfg["budget"],source_manifest=manifest)
        role_laws=[]
        for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
            role_data=data["fit"] if role=="OSF_DEFENSE_FIT" else load_allowed_role(manifest,cfg["seed"],role)
            w=role_data["wire"];e1,e2,mass=policy.law_codes(w["code_r1"],w["code_r2"])
            law_sha=write_npz(private/arm_id/(role+"_LAW.npz"),{"row_id":w["row_id"],"unit":w["unit"],"code_r1":e1,"code_r2":e2,"mass":mass})
            role_laws.append({"role":role,"rows":len(mass),"law_sha256":law_sha})
        inner=load_allowed_role(manifest,cfg["seed"],"INNER_SELECTION",fields=("income","occupation","references"))
        w=inner["wire"];a,b,mass=policy.law_codes(w["code_r1"],w["code_r2"])
        ip={"code_r1":a,"code_r2":b,"b0_code_r1":w["code_r1"],"b0_code_r2":w["code_r2"]}
        utility=backend.expected_rows(ip,mass,space,(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
        gates=[];utility_arrays={"row_id":w["row_id"],"unit":w["unit"]}
        for ref,values in utility["excess"].items():
            for key,value in values.items():
                metric=key.rsplit("_",1)[1];cap=ORIGINAL_CAPS[metric];margin=float(value.mean())
                gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
                utility_arrays[f"{ref}_{key}"]=value
        write_npz(private/arm_id/"INNER_UTILITY_ROWS.npz",utility_arrays)
        change=float(law.change_coefficients()[0]+law.change_coefficients()[1]@theta)
        record.update(arm_id=arm_id,fitted_objective=selected["objective"],fitted_recovery=selected["views"],
                      actual_change_rate=change,minimum_fitting_constraint_margin=float(margins.min()),
                      optimization_deployment_max_gap=gap,policy_receipt_sha256=receipt_sha,
                      policy_package_sha256=package["package_sha256"],parameter_count=package["parameter_count"],
                      selected_coordinate_parameters=int(law.parameters),role_laws=role_laws,
                      inner_point_utility=gates,inner_point_eligible=all(x["pass"] for x in gates))
        decision_views={}
        decisions=(space.probability1[c1].argmax(1),space.probability2[c2].argmax(1))
        for name,key in (("r1",decisions[0][:,None]),("r2",decisions[1][:,None]),("pair",np.c_[*decisions])):
            unique,inverse=np.unique(key,axis=0,return_inverse=True)
            count=np.stack([np.bincount(inverse,weights=(sex==s)/len(sex),minlength=len(unique)) for s in (0,1)])
            decision_views[name]=float(count.max(0).sum())
        lower=decision_views["pair"]+.5*(decision_views["r1"]+decision_views["r2"])
        record.update(decision_only_relaxation_bound=lower,heuristic_global_gap_upper_bound=max(0.,selected["objective"]-lower),
                      exact_remaining_global_gap=None)
    public=Path(cfg["public_output"]);public.mkdir(parents=True,exist_ok=True)
    write_once(public/(cfg["unit_id"]+"_RECEIPT.json"),record,False)
    return {"unit_id":cfg["unit_id"],"status":record["status"],"solver_calls":len(archives)}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();print(json.dumps(run(args.config)))
