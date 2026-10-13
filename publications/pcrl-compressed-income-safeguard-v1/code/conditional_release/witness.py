"""Serialize the exact pre-protection minimum-NLL witness as matched control."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .source import load_manifest,load_allowed_role,read_exact,sha_file,write_once
from .fit import (prepare_inputs,write_npz,AllocationPolicy,save_allocation_policy)
from .secondary import (load_template_policy,TemplatePolicy,save_template_policy,bank_expected_rows)
from .preparation import expected_rows,ORIGINAL_CAPS


def run(config):
    from .scheduler import assert_active_from_env
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text())
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("witness materialization registration differs")
    manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    data=prepare_inputs(manifest,cfg["seed"])
    # Archive contains only allowed solver quantities, no raw person arrays.
    from zipfile import ZipFile
    item={"path":cfg["minimum_nll_archive"],"sha256":cfg["minimum_nll_archive_sha256"]}
    with ZipFile(item["path"]) as z:keys={name[:-4] for name in z.namelist()}
    arrays=read_exact(item,keys)
    if "theta" not in arrays:raise PermissionError("utility witness did not yield feasible coefficients")
    theta=arrays["theta"]
    target=Path(cfg["private_output"])
    if cfg["kind"]=="allocation":
        if np.min(theta)<-1e-12 or np.max(theta)>1+1e-12:raise ValueError("witness allocation bounds")
        policy=AllocationPolicy(data["historical"],data["context"],np.clip(theta,0,1),cfg["mode"])
        receipt,receipt_sha=save_allocation_policy(target,policy,seed=cfg["seed"],arm_id="UT_SELECTED",budget=cfg["budget"],source_manifest=manifest)
    elif cfg["kind"]=="template":
        donor=load_template_policy(cfg["candidate_policy_directory"],manifest,expected_receipt_sha256=cfg["candidate_policy_receipt_sha256"])
        policy=TemplatePolicy(donor.space,donor.context_map,theta,donor.metadata)
        receipt,receipt_sha=save_template_policy(target,policy,seed=cfg["seed"],arm_id="UT_SELECTED",budget=cfg["budget"],manifest=manifest)
    else:raise PermissionError("witness family unregistered")
    role_laws=[];exact_alias=True
    for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
        source=data["fit"] if role=="OSF_DEFENSE_FIT" else load_allowed_role(manifest,cfg["seed"],role)
        w=source["wire"];a,b,mass=policy.law_codes(w["code_r1"],w["code_r2"])
        own=(a==w["code_r1"][:,None])&(b==w["code_r2"][:,None])
        alias=bool(np.all(mass[~own]==0)&np.all(mass.sum(1)==1))
        exact_alias&=alias
        law_sha=write_npz(target/(role+"_LAW.npz"),{"row_id":w["row_id"],"unit":w["unit"],"code_r1":a,"code_r2":b,"mass":mass})
        role_laws.append({"role":role,"law_sha256":law_sha,"rows":len(mass),"exact_p0_alias":alias})
    inner=load_allowed_role(manifest,cfg["seed"],"INNER_SELECTION",fields=("income","occupation","references"))
    w=inner["wire"];a,b,mass=policy.law_codes(w["code_r1"],w["code_r2"])
    prepared={"code_r1":a,"code_r2":b,"b0_code_r1":w["code_r1"],"b0_code_r2":w["code_r2"]}
    utility=bank_expected_rows(prepared,mass,data["space"],(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
    gates=[];uarrays={"row_id":w["row_id"],"unit":w["unit"]}
    for ref,values in utility["excess"].items():
        for key,value in values.items():
            cap=ORIGINAL_CAPS[key.rsplit("_",1)[1]];margin=float(value.mean())
            gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
            uarrays[f"{ref}_{key}"]=value
    write_npz(target/"INNER_UTILITY_ROWS.npz",uarrays)
    result={"schema":"CBR1-exact-utility-witness-control-v1","unit_id":cfg["unit_id"],"source_seed":cfg["seed"],"arm_id":"UT_SELECTED",
            "status":"COMPLETE","witness_archive_sha256":cfg["minimum_nll_archive_sha256"],"new_solver_calls":0,
            "kind":cfg["kind"],"budget":cfg["budget"],"policy_receipt_sha256":receipt_sha,"policy_package_sha256":receipt["package_sha256"],
            "role_laws":role_laws,"exact_p0_alias":exact_alias,"inner_point_utility":gates,"inner_point_eligible":all(x["pass"] for x in gates),
            "inner_sex_access":False,"fit_sex_labels_read":False}
    write_once(cfg["public_output"],result,False)
    print(json.dumps({"status":"COMPLETE","unit_id":cfg["unit_id"],"eligible":result["inner_point_eligible"],"p0_alias":exact_alias}))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args();run(args.config)

if __name__=="__main__":main()
