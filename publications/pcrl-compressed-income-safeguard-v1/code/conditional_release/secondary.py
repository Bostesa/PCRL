"""Registered support/capacity package comparisons after primary verification."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from .source import (load_manifest,load_allowed_role,load_historical_policy,sha_file,write_once,read_exact)
from .fit import prepare_inputs,archive_result,write_npz,witnesses,fitting_constraint_receipt
from .contexts import ContextMap
from .templates import build_template_law,template_probabilities
from .preparation import task_costs,constraint_rows,expected_rows,ORIGINAL_CAPS
from .reader_extension import frozen_local_loss_tables,losses_for_actions,replay_loss_parity
from .lp import solve_protection,direct_bayes,certify_infeasibility


def bank_action_costs(space,prepared,labels):
    """Frozen bank loss lookup avoids any person×action×class tensor."""
    from .preparation import probability_costs
    result={}
    for r,y in enumerate(labels,1):
        bank=getattr(space,f"probability{r}");codes=prepared[f"code_r{r}"]
        table={metric:np.empty((len(bank),bank.shape[1]),np.float64) for metric in ("accuracy","nll","brier")}
        for label in range(bank.shape[1]):
            losses=probability_costs(bank,np.full(len(bank),label,np.int64))
            for metric,value in losses.items():table[metric][:,label]=value
        for metric,lookup in table.items():result[f"r{r}_{metric}"]=lookup[codes,y[:,None]]
    return result


def bank_expected_rows(prepared,mass,space,labels,references):
    from .preparation import reference_costs
    costs=bank_action_costs(space,prepared,labels);refs=reference_costs(references,labels)
    expected={key:(mass*value).sum(1) for key,value in costs.items()}
    return {"expected":expected,"reference":refs,
            "excess":{ref:{key:expected[key]-value for key,value in values.items()} for ref,values in refs.items()}}


@dataclass
class TemplatePolicy:
    space: object
    context_map: ContextMap
    theta: np.ndarray
    metadata: dict

    def law_codes(self,c1,c2):
        context=self.context_map.apply(self.space.probability1[c1],self.space.probability2[c2])
        return template_probabilities(self.metadata,self.theta,self.space,c1,c2,context)


def save_template_policy(directory,policy,*,seed,arm_id,budget,manifest):
    directory=Path(directory)
    if directory.exists() or ".private" not in directory.parts:raise PermissionError("new private templatepolicy required")
    directory.mkdir(parents=True,mode=0o700)
    context=policy.context_map
    arrays={"theta":policy.theta,"context_mapping":context.mapping,"raw_fit_counts":context.raw_fit_counts,"merged_fit_counts":context.merged_fit_counts}
    package_sha=write_npz(directory/"template.npz",arrays)
    receipt={"schema":"CBR1-template-policy-v1","seed":seed,"arm_id":arm_id,"budget":budget,
             "package_sha256":package_sha,"metadata":policy.metadata,"context_labels":list(context.labels),
             "public_bank_sha256":manifest["seeds"][str(seed)]["public_bank"]["sha256"]}
    return receipt,write_once(directory/"POLICY_RECEIPT.json",receipt)


def load_template_policy(directory,manifest,*,expected_receipt_sha256=None):
    directory=Path(directory);p=directory/"POLICY_RECEIPT.json"
    if expected_receipt_sha256 and sha_file(p)!=expected_receipt_sha256:raise PermissionError("template receipt hash differs")
    rec=json.loads(p.read_text())
    if rec["schema"]!="CBR1-template-policy-v1" or rec["public_bank_sha256"]!=manifest["seeds"][str(rec["seed"])]["public_bank"]["sha256"]:
        raise PermissionError("template source differs")
    arrays=read_exact({"path":str(directory/"template.npz"),"sha256":rec["package_sha256"]},{"theta","context_mapping","raw_fit_counts","merged_fit_counts"})
    context=ContextMap(arrays["context_mapping"],arrays["raw_fit_counts"],arrays["merged_fit_counts"],tuple(rec["context_labels"]))
    return TemplatePolicy(load_historical_policy(manifest,rec["seed"],"V_J").space,context,arrays["theta"],rec["metadata"])


def run(config):
    from .scheduler import assert_active_from_env
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text())
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("secondary lock/unit differs")
    # Registered dependent stage: primary receipt must exist and all eligible
    # primary protection units must complete before this support package.
    primary=json.loads(Path(cfg["primary_receipt"]).read_text())
    if primary["status"]!="COMPLETE" or primary["seed"]!=cfg["seed"]:
        raise PermissionError("primary prerequisites incomplete")
    independent=Path(cfg["independent_primary_receipt"])
    if sha_file(independent)!=cfg["independent_primary_receipt_sha256"]:
        raise PermissionError("independent primary replay receipt changed")
    verification=json.loads(independent.read_text())
    if not str(verification.get("status","")).startswith("PASS"):
        raise PermissionError("primary scientific replay did not pass")
    manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    data=prepare_inputs(manifest,cfg["seed"])
    prep=json.loads(Path(cfg["preparation_report"]).read_text())
    if prep["context_report"]!=data["context"].report():raise PermissionError("secondary context changed")
    private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    # Source frozen safeguards require FIT SEX to evaluate original models on
    # WIDE actions. This is registered guard evaluation, not reader fitting or
    # protection objective optimization before utility witnesses.
    sex=load_allowed_role(manifest,cfg["seed"],"OSF_DEFENSE_FIT",fields=("sex",))["sex"]["sex"]
    tables,catalog,arrays,index=frozen_local_loss_tables(manifest,cfg["seed"],data["space"],data["fit"]["fold"],sex)
    guard_parity=replay_loss_parity(tables,arrays,index,data["prepared"],data["fit"]["fold"],sex)
    del arrays
    # One source seed, one support per unit, both budgets. This bounds WIDE
    # allocation before any solve and frees memory between distinct units.
    support=cfg["support"]
    wire=data["fit"]["wire"]
    law,metadata=build_template_law(data["space"],wire["code_r1"],wire["code_r2"],data["contexts"],support,cfg["maximum_input_bytes"])
    p={"code_r1":law.emitted1,"code_r2":law.emitted2,"b0_code_r1":law.identity1,"b0_code_r2":law.identity2}
    costs=bank_action_costs(data["space"],p,data["labels"])
    losses=(data["reader_losses"] if support=="NARROW" else losses_for_actions(tables,law.emitted1,law.emitted2,data["fit"]["fold"],sex))
    setups=[]
    for budget in ("STRICT","ORIGINAL"):
        constraints,translations=constraint_rows(law,costs,data["fit"]["fold"],data["references"],losses,budget)
        witness_data={"costs":costs}
        minimum,maximum=witnesses(law,constraints,witness_data,time_limit=cfg["solve_time_limit_s"])
        name=f"{support}_{budget}"
        receipts={"minimum_nll":archive_result(private,name+"_MIN_NLL",minimum),"maximum_actual_change":archive_result(private,name+"_MAX_CHANGE",maximum)}
        phase=None
        if not maximum["success"] and maximum["status"]==2:
            attempt=certify_infeasibility(maximum["program"],time_limit=cfg["solve_time_limit_s"])
            phase=archive_result(private,name+"_PHASEI",attempt);phase["infeasibility_certified"]=attempt["infeasibility_certified"]
        setups.append({"budget":budget,"constraints":constraints,"minimum":minimum,"maximum":maximum,"witnesses":receipts,"infeasibility":phase})
    results=[]
    for setup in setups:
        budget=setup["budget"];arm_id=("N" if support=="NARROW" else "W")+"_"+budget
        rec={"arm_id":arm_id,"budget":budget,"support":support,"source_seed":cfg["seed"],"witnesses":setup["witnesses"],"infeasibility":setup["infeasibility"]}
        if not setup["maximum"]["success"]:
            rec["status"]="UNTESTED_DEPENDENCY_INFEASIBLE" if setup["infeasibility"] and setup["infeasibility"]["infeasibility_certified"] else "UNTESTED_DEPENDENCY_UNRESOLVED"
            results.append(rec);continue
        assert_active_from_env(stage="protection")
        result=solve_protection(law,sex,setup["constraints"],time_limit=cfg["solve_time_limit_s"])
        rec["protection"]=archive_result(private,arm_id+"_PROTECTION",result)
        if not result["success"]:
            rec["status"]="INCOMPLETE_PROTECTION_SOLVE";results.append(rec);continue
        theta=result["theta"]
        policy=TemplatePolicy(data["space"],data["context"],theta,metadata)
        q=law.probabilities(theta);e1,e2,deploy=policy.law_codes(wire["code_r1"],wire["code_r2"])
        if not np.array_equal(e1,law.emitted1) or not np.array_equal(e2,law.emitted2) or not np.allclose(q,deploy,rtol=0,atol=1e-14):
            raise RuntimeError("template optimizer/deployment law differs")
        objective,views_=direct_bayes(law,theta,sex)
        p_rec,p_sha=save_template_policy(private/arm_id,policy,seed=cfg["seed"],arm_id=arm_id,budget=budget,manifest=manifest)
        role_laws=[]
        for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
            source=data["fit"] if role=="OSF_DEFENSE_FIT" else load_allowed_role(manifest,cfg["seed"],role)
            w=source["wire"];a,b,mass=policy.law_codes(w["code_r1"],w["code_r2"])
            law_sha=write_npz(private/arm_id/(role+"_LAW.npz"),{"row_id":w["row_id"],"unit":w["unit"],"code_r1":a,"code_r2":b,"mass":mass})
            role_laws.append({"role":role,"rows":len(mass),"law_sha256":law_sha})
        inner=load_allowed_role(manifest,cfg["seed"],"INNER_SELECTION",fields=("income","occupation","references"))
        w=inner["wire"];a,b,mass=policy.law_codes(w["code_r1"],w["code_r2"])
        ip={"code_r1":a,"code_r2":b,"b0_code_r1":w["code_r1"],"b0_code_r2":w["code_r2"]}
        utility=bank_expected_rows(ip,mass,data["space"],(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
        gates=[];utility_arrays={"row_id":w["row_id"],"unit":w["unit"]}
        for ref,values in utility["excess"].items():
            for key,value in values.items():
                metric=key.rsplit("_",1)[1];cap=ORIGINAL_CAPS[metric];margin=float(value.mean())
                gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
                utility_arrays[f"{ref}_{key}"]=value
        write_npz(private/arm_id/"INNER_UTILITY_ROWS.npz",utility_arrays)
        max_change=float(law.change_coefficients()[0]+law.change_coefficients()[1]@setup["maximum"]["theta"])
        change=float(law.change_coefficients()[0]+law.change_coefficients()[1]@theta)
        rec.update({"status":"COMPLETE","fitted_objective":objective,"fitted_recovery":views_,"fitting_constraints":fitting_constraint_receipt(setup["constraints"],theta),"actual_change_rate":change,
                    "maximum_actual_change":max_change,"policy_receipt_sha256":p_sha,"policy_package_sha256":p_rec["package_sha256"],
                    "role_laws":role_laws,"inner_point_utility":gates,"inner_point_eligible":bool(all(g["pass"] for g in gates)),
                    "parameter_count":law.parameters,"sameclass_support":True,"full_declared_support_columns":True})
        results.append(rec)
    report={"schema":"CBR1-R3-secondary-unit-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],"support":support,"status":"COMPLETE",
            "science_lock_sha256":cfg["science_lock_sha256"],"source_manifest_sha256":cfg["source_manifest_sha256"],
            "primary_receipt_sha256":sha_file(cfg["primary_receipt"]),"independent_primary_receipt_sha256":cfg["independent_primary_receipt_sha256"],"support_metadata":metadata,"frozen_guard_parity":guard_parity,
            "setups":results,"inner_sex_access":False,"fresh_reader_fits":0,
            "lp_solve_calls":sum(2+int(s["infeasibility"] is not None)+int("protection" in r) for s,r in zip(setups,results))}
    public=Path(cfg["public_output"]);public.mkdir(parents=True,exist_ok=True)
    write_once(public/(cfg["unit_id"]+"_RECEIPT.json"),report,False)
    print(json.dumps({"status":"COMPLETE","unit_id":cfg["unit_id"],"arms":sum(r["status"]=="COMPLETE" for r in results)}))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args();run(args.config)

if __name__=="__main__":main()
