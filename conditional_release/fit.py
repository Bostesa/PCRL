"""Primary frozen-donor allocation fits and deployed law serialization."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os
from pathlib import Path
import numpy as np
from .contexts import ContextMap, fit_context_map
from .source import load_allowed_role, load_historical_policy, load_bank, donor_law, write_once, sha_file
from .preparation import (task_costs, reference_costs, restore_local_reader_losses,
                          constraint_rows, expected_rows)
from .lp import build_mixture_problem, solve_linear, solve_protection, direct_bayes


def write_npz(path, arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return sha_file(path)


@dataclass
class AllocationPolicy:
    donor_policy: object
    context_map: ContextMap
    theta: np.ndarray
    mode: str

    def __post_init__(self):
        self.theta=np.asarray(self.theta,np.float64).copy()
        expected=1 if self.mode=="scalar" else max(1,len(self.context_map.labels))
        if self.mode not in ("scalar","conditional") or self.theta.shape!=(expected,) or np.any(self.theta<0) or np.any(self.theta>1):
            raise ValueError("allocation deployment coefficient invalid")
        if self.donor_policy.arm!="V_J":raise ValueError("frozen V-J donor required")

    @property
    def space(self):return self.donor_policy.space

    def law_codes(self,c1,c2):
        prepared=self.space.prepare(c1,c2)
        return prepared["code_r1"],prepared["code_r2"],self.law_prepared(prepared)

    def law_prepared(self,p):
        from incumbent_view.r2_policy import Policy
        c=self.context_map.apply(self.space.probability1[p["b0_code_r1"]],self.space.probability2[p["b0_code_r2"]])
        coefficient=np.zeros(len(c),np.float64);supported=c>=0
        coefficient[supported]=self.theta[0] if self.mode=="scalar" else self.theta[c[supported]]
        donor=Policy(self.space,"V_J",self.donor_policy.weights,1.,1.).law_prepared(p)
        mass=donor*coefficient[:,None];mass[:,0]+=1-coefficient
        return mass


def save_allocation_policy(directory,policy,*,seed,arm_id,budget,source_manifest):
    directory=Path(directory)
    if directory.exists() or ".private" not in directory.parts:raise PermissionError("new private policy required")
    directory.mkdir(parents=True,mode=0o700)
    arrays={"theta":policy.theta,"context_mapping":policy.context_map.mapping,
            "raw_fit_counts":policy.context_map.raw_fit_counts,"merged_fit_counts":policy.context_map.merged_fit_counts}
    package_sha=write_npz(directory/"allocation.npz",arrays)
    rec={"schema":"CBR1-allocation-policy-v1","seed":seed,"arm_id":arm_id,"budget":budget,"mode":policy.mode,
         "package_sha256":package_sha,"context_labels":list(policy.context_map.labels),
         "frozen_donor_receipt_sha256":source_manifest["seeds"][str(seed)]["policies"]["V_J"]["receipt"]["sha256"],
         "public_bank_sha256":source_manifest["seeds"][str(seed)]["public_bank"]["sha256"]}
    return rec,write_once(directory/"POLICY_RECEIPT.json",rec)


def load_allocation_policy(directory,manifest,*,expected_receipt_sha256=None):
    directory=Path(directory);receipt=directory/"POLICY_RECEIPT.json"
    if expected_receipt_sha256 and sha_file(receipt)!=expected_receipt_sha256:raise PermissionError("allocation receipt changed")
    rec=json.loads(receipt.read_text())
    if rec["schema"]!="CBR1-allocation-policy-v1" or sha_file(directory/"allocation.npz")!=rec["package_sha256"]:
        raise PermissionError("allocation package differs")
    from .source import read_exact
    arrays=read_exact({"path":str(directory/"allocation.npz"),"sha256":rec["package_sha256"]},
                     {"theta","context_mapping","raw_fit_counts","merged_fit_counts"})
    seed=rec["seed"]
    if (manifest["seeds"][str(seed)]["policies"]["V_J"]["receipt"]["sha256"]!=rec["frozen_donor_receipt_sha256"]
        or manifest["seeds"][str(seed)]["public_bank"]["sha256"]!=rec["public_bank_sha256"]):
        raise PermissionError("allocation donor/bank binding differs")
    context=ContextMap(arrays["context_mapping"],arrays["raw_fit_counts"],arrays["merged_fit_counts"],tuple(rec["context_labels"]))
    return AllocationPolicy(load_historical_policy(manifest,seed,"V_J"),context,arrays["theta"],rec["mode"])


def prepare_inputs(manifest,seed):
    fit=load_allowed_role(manifest,seed,"OSF_DEFENSE_FIT",fields=("income","occupation","references"))
    wire=fit["wire"];historical=load_historical_policy(manifest,seed,"V_J");space=historical.space
    p=space.prepare(wire["code_r1"],wire["code_r2"])
    context=fit_context_map(wire["prob_r1"],wire["prob_r2"])
    donor=donor_law(manifest,seed,wire["code_r1"],wire["code_r2"])
    restored=historical.law_prepared(p)
    reconstructed=donor*historical.eta1;reconstructed[:,0]+=1-historical.eta1
    if not np.array_equal(restored,reconstructed):raise PermissionError("frozen donor direct reconstruction does not exactly match selected source law")
    ids=context.apply(wire["prob_r1"],wire["prob_r2"])
    donor[ids<0]=0;donor[ids<0,0]=1
    labels=(fit["income"]["income"],fit["occupation"]["occupation"])
    cost=task_costs(space,p,labels);refs=reference_costs(fit["references"],labels)
    reader_losses,reader_catalog=restore_local_reader_losses(manifest,seed)
    return {"fit":fit,"space":space,"prepared":p,"historical":historical,"context":context,"contexts":ids,
            "donor":donor,"labels":labels,"costs":cost,"references":refs,"reader_losses":reader_losses,
            "reader_catalog":reader_catalog,"donor_exact_parity":True}


def mixture_setup(data,mode,budget):
    p=data["prepared"]
    law=build_mixture_problem(p["code_r1"],p["code_r2"],p["b0_code_r1"],p["b0_code_r2"],data["donor"],
                              np.maximum(data["contexts"],0),coefficient_mode=mode)
    constraints,translations=constraint_rows(law,data["costs"],data["fit"]["fold"],data["references"],data["reader_losses"],budget)
    return law,constraints,translations


def fitting_constraint_receipt(constraints,theta,numeric_envelope=1e-7):
    """Finite constraints, with numerical boundary distinct from cap passage."""
    values=np.asarray(constraints.matrix@theta).ravel();slack=constraints.rhs-values
    clauses=[]
    for name,lhs,rhs,margin in zip(constraints.names,values,constraints.rhs,slack):
        label=("PASS" if margin>=numeric_envelope else "VIOLATION" if margin < -numeric_envelope else "NUMERICAL_BOUNDARY_UNRESOLVED")
        clauses.append({"name":name,"affine_lhs":float(lhs),"affine_rhs":float(rhs),"slack":float(margin),"status":label})
    return {"numerical_envelope":numeric_envelope,"scientific_caps_unchanged":True,"clauses":clauses,
            "minimum_slack":float(slack.min()),"violation_count":sum(c["status"]=="VIOLATION" for c in clauses),
            "boundary_unresolved_count":sum(c["status"]=="NUMERICAL_BOUNDARY_UNRESOLVED" for c in clauses)}


def scalar_result(result):
    return {key:value for key,value in result.items() if key in ("success","status","message","certificate")}


def archive_result(directory,name,result):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    from scipy import sparse
    arrays={key:result[key] for key in ("theta","primal","inequality_dual","equality_dual","lower_dual","upper_dual") if result.get(key) is not None}
    program=result.get("program")
    if program:
        arrays.update({"objective":program.objective,"inequality_rhs":program.inequality_rhs,
                       "equality_rhs":program.equality_rhs,"bounds":program.bounds})
        for prefix,matrix in (("inequality",program.inequality_matrix),("equality",program.equality_matrix)):
            matrix=sparse.csr_matrix(matrix)
            arrays.update({f"{prefix}_data":matrix.data,f"{prefix}_indices":matrix.indices,
                           f"{prefix}_indptr":matrix.indptr,f"{prefix}_shape":np.asarray(matrix.shape,np.int64)})
    sha=write_npz(directory/f"{name}.npz",arrays)
    return {**scalar_result(result),"archive_sha256":sha,"archive_name":f"{name}.npz",
            "constraint_names":list(program.names) if program else [],"objective_constant":program.objective_constant if program else None}


def witnesses(law,constraints,data,*,time_limit=120.):
    nll=data["costs"]["r1_nll"]+data["costs"]["r2_nll"]
    intercept,coefficient=law.expected_coefficients(nll)
    minimum=solve_linear(law,constraints,coefficient,objective_constant=intercept,time_limit=time_limit)
    change_intercept,change_coefficient=law.change_coefficients()
    maximum=solve_linear(law,constraints,change_coefficient,objective_constant=change_intercept,maximize=True,time_limit=time_limit)
    return minimum,maximum


def prep_worker(config):
    from .scheduler import assert_active_from_env
    from .source import load_manifest
    active=assert_active_from_env(stage="admission")
    cfg=json.loads(Path(config).read_text());manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    if active["unit_id"]!=cfg["unit_id"]:raise PermissionError("prep unit differs")
    data=prepare_inputs(manifest,cfg["seed"]);context=data["context"]
    private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    p=data["prepared"]
    arrays={"context_mapping":context.mapping,"raw_fit_counts":context.raw_fit_counts,"merged_fit_counts":context.merged_fit_counts,
            "contexts":data["contexts"],"donor":data["donor"],"fold":data["fit"]["fold"],
            **{k:p[k] for k in ("code_r1","code_r2","b0_code_r1","b0_code_r2","valid")}}
    array_sha=write_npz(private/"PREPARATION.npz",arrays)
    translations=[];identities={}
    for mode in ("scalar","conditional"):
        for budget in ("STRICT","ORIGINAL"):
            law,constraints,translation=mixture_setup(data,mode,budget)
            identities[f"{mode}/{budget}"]={"parameters":law.parameters,"constraint_count":len(constraints.names),
                                           "constraint_names":list(constraints.names),"identity_feasible":bool(np.all(constraints.rhs>=0))}
            if mode=="scalar":translations.extend(translation)
    import csv
    public=Path(cfg["public_output"]);public.mkdir(parents=True,exist_ok=True)
    with (public/f"R3_S{cfg['seed']}_BUDGET_TRANSLATION.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(translations[0]));writer.writeheader();writer.writerows(translations)
    report={"schema":"CBR1-R3-preparation-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],"status":"COMPLETE",
            "source_manifest_sha256":cfg["source_manifest_sha256"],"donor_exact_parity":True,
            "context_report":context.report(),"local_reader_restore":data["reader_catalog"],
            "setup_dimensions_and_identity":identities,"private_preparation_sha256":array_sha,
            "context_labels":list(context.labels),"fresh_sex_labels_read":False,
            "fit_task_label_files_read":2,"fit_reference_files_read":1,"solver_calls":0,
            "inner_labels_read":False,"closed_role_reads":False}
    write_once(public/f"R3_S{cfg['seed']}_PREPARATION.json",report,False)
    print(json.dumps({"status":"COMPLETE","unit_id":cfg["unit_id"],"contexts":len(context.labels)}))


def primary_worker(config):
    from .scheduler import assert_active_from_env
    from .source import load_manifest
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text());manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("science unit/lock differs")
    data=prepare_inputs(manifest,cfg["seed"])
    prep=json.loads(Path(cfg["preparation_report"]).read_text())
    if data["context"].report()!=prep["context_report"]:raise PermissionError("frozen input-only context changed")
    private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    all_setups=[]
    from .lp import certify_infeasibility
    # All utility-only witnesses complete before opening this unit's FIT SEX.
    for budget in ("STRICT","ORIGINAL"):
        for mode in ("scalar","conditional"):
            assert_active_from_env(stage="protection")
            law,constraints,translations=mixture_setup(data,mode,budget)
            minimum,maximum=witnesses(law,constraints,data,time_limit=cfg["solve_time_limit_s"])
            name=f"{budget}_{mode}"
            receipts={"minimum_nll":archive_result(private,name+"_MIN_NLL",minimum),
                      "maximum_actual_change":archive_result(private,name+"_MAX_CHANGE",maximum)}
            infeasibility=None
            if not maximum["success"] and maximum["status"]==2:
                phase=certify_infeasibility(maximum["program"],time_limit=cfg["solve_time_limit_s"])
                infeasibility=archive_result(private,name+"_PHASEI",phase)
                infeasibility["infeasibility_certified"]=phase["infeasibility_certified"]
            all_setups.append({"budget":budget,"mode":mode,"law":law,"constraints":constraints,
                               "minimum":minimum,"maximum":maximum,"witness_receipts":receipts,
                               "infeasibility":infeasibility,"translations":translations})
    sex=load_allowed_role(manifest,cfg["seed"],"OSF_DEFENSE_FIT",fields=("sex",))["sex"]["sex"]
    public=[];frontier=[]
    for setup in all_setups:
        budget,mode=setup["budget"],setup["mode"];name=f"{budget}_{mode}";law=setup["law"]
        max_=setup["maximum"]
        change=None if not max_["success"] else float(sum(x*y for x,y in zip(law.change_coefficients()[1],max_["theta"]))+law.change_coefficients()[0])
        rec={"budget":budget,"mode":mode,"witnesses":setup["witness_receipts"],"infeasibility":setup["infeasibility"],
             "parameters":law.parameters,"constraint_count":len(setup["constraints"].names),"maximum_actual_change":change,
             "donor_exact_parity":True,"context_report_sha256":hashlib.sha256(json.dumps(data["context"].report(),sort_keys=True).encode()).hexdigest()}
        if not max_["success"]:
            rec["status"]="UNTESTED_DEPENDENCY_INFEASIBLE" if setup["infeasibility"] and setup["infeasibility"]["infeasibility_certified"] else "UNTESTED_DEPENDENCY_UNRESOLVED"
            public.append(rec);continue
        assert_active_from_env(stage="protection")
        result=solve_protection(law,sex,setup["constraints"],time_limit=cfg["solve_time_limit_s"])
        rec["protection"]=archive_result(private,name+"_PROTECTION",result)
        if not result["success"]:
            rec["status"]="INCOMPLETE_PROTECTION_SOLVE";public.append(rec);continue
        theta=result["theta"]
        # HiGHS may return boundary noise <=1e-12; preserve primal archive and
        # make executable parameters exact [0,1], independently replay margins.
        if np.min(theta)<-1e-12 or np.max(theta)>1+1e-12:raise RuntimeError("allocation parameter out of deployment bounds")
        theta=np.clip(theta,0,1)
        policy=AllocationPolicy(data["historical"],data["context"],theta,mode)
        q=law.probabilities(theta);check=policy.law_prepared(data["prepared"])
        if not np.allclose(q,check,atol=1e-14,rtol=0):raise RuntimeError("optimization/deployment law mismatch")
        objective,views_=direct_bayes(law,theta,sex)
        actual_change=float(law.change_coefficients()[0]+law.change_coefficients()[1]@theta)
        arm_id=("G" if mode=="scalar" else "C")+"_"+budget
        p_receipt,p_sha=save_allocation_policy(private/arm_id,policy,seed=cfg["seed"],arm_id=arm_id,budget=budget,source_manifest=manifest)
        role_laws=[]
        for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
            source=data["fit"] if role=="OSF_DEFENSE_FIT" else load_allowed_role(manifest,cfg["seed"],role)
            wire=source["wire"];e1,e2,mass=policy.law_codes(wire["code_r1"],wire["code_r2"])
            role_sha=write_npz(private/arm_id/(role+"_LAW.npz"),{"row_id":wire["row_id"],"unit":wire["unit"],"code_r1":e1,"code_r2":e2,"mass":mass})
            role_laws.append({"role":role,"law_sha256":role_sha,"rows":len(mass)})
        inner=load_allowed_role(manifest,cfg["seed"],"INNER_SELECTION",fields=("income","occupation","references"))
        ip=data["space"].prepare(inner["wire"]["code_r1"],inner["wire"]["code_r2"]);iq=policy.law_prepared(ip)
        utility=expected_rows(ip,iq,data["space"],(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
        from .preparation import ORIGINAL_CAPS
        gates=[];private_utility={}
        for ref,values in utility["excess"].items():
            for key,value in values.items():
                metric=key.rsplit("_",1)[1];margin=float(value.mean());cap=ORIGINAL_CAPS[metric]
                gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
                private_utility[f"{ref}_{key}"]=value
        write_npz(private/arm_id/"INNER_UTILITY_ROWS.npz",{"row_id":inner["wire"]["row_id"],"unit":inner["wire"]["unit"],**private_utility})
        rec.update({"status":"COMPLETE","arm_id":arm_id,"theta_sha256":hashlib.sha256(theta.tobytes()).hexdigest(),"theta_range":[float(theta.min()),float(theta.max())],"fitted_objective":objective,
                    "fitted_recovery":views_,"fitting_constraints":fitting_constraint_receipt(setup["constraints"],theta),"actual_change_rate":actual_change,"policy_receipt_sha256":p_sha,
                    "policy_package_sha256":p_receipt["package_sha256"],"role_laws":role_laws,
                    "inner_point_utility":gates,"inner_point_eligible":bool(all(x["pass"] for x in gates)),
                    "source_seed":cfg["seed"]})
        public.append(rec)
        frontier.append({"seed":cfg["seed"],"arm_id":arm_id,"budget":budget,"mode":mode,"fitted_objective":objective,
                         "actual_change_rate":actual_change,"maximum_feasible_change":change,"inner_point_eligible":rec["inner_point_eligible"]})
    report={"schema":"CBR1-R3-primary-unit-v1","unit_id":cfg["unit_id"],"seed":cfg["seed"],"status":"COMPLETE",
            "science_lock_sha256":cfg["science_lock_sha256"],"source_manifest_sha256":cfg["source_manifest_sha256"],
            "preparation_report_sha256":sha_file(cfg["preparation_report"]),"setups":public,
            "inner_sex_access":False,"closed_role_access":False,"fresh_reader_fits":0,
            "lp_solve_calls":sum(2+int(s["infeasibility"] is not None)+int("protection" in p) for s,p in zip(all_setups,public))}
    public_root=Path(cfg["public_output"]);public_root.mkdir(parents=True,exist_ok=True)
    write_once(public_root/(cfg["unit_id"]+"_RECEIPT.json"),report,False)
    import csv
    if frontier:
        with (public_root/(cfg["unit_id"]+"_FRONTIER.csv")).open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(frontier[0]));writer.writeheader();writer.writerows(frontier)
    print(json.dumps({"unit_id":cfg["unit_id"],"status":"COMPLETE","arms":len(frontier),"lp_solve_calls":report["lp_solve_calls"]}))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("preparation","primary"),required=True);parser.add_argument("--config",required=True)
    args=parser.parse_args();(prep_worker if args.phase=="preparation" else primary_worker)(args.config)

if __name__=="__main__":main()
