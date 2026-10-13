"""Restore P0 and the exact historical policies without new scientific fits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .source import load_manifest,load_allowed_role,load_historical_policy,sha_file,write_once
from .fit import write_npz
from .preparation import expected_rows,ORIGINAL_CAPS


def run(config):
    from .scheduler import assert_active_from_env
    active=assert_active_from_env(stage="protection")
    cfg=json.loads(Path(config).read_text())
    if active["unit_id"]!=cfg["unit_id"] or sha_file(cfg["science_lock"])!=cfg["science_lock_sha256"]:
        raise PermissionError("baseline registration differs")
    manifest=load_manifest(cfg["source_manifest"],cfg["source_manifest_sha256"])
    seed=cfg["seed"];private=Path(cfg["private_output"]);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    policies={"HIST_V_J":load_historical_policy(manifest,seed,"V_J"),
              "HIST_V_S21":load_historical_policy(manifest,seed,"V_S21")}
    space=policies["HIST_V_J"].space
    inner=load_allowed_role(manifest,seed,"INNER_SELECTION",fields=("income","occupation","references"))
    arms=[]
    for arm in ("P0","HIST_V_J","HIST_V_S21"):
        directory=private/arm;directory.mkdir(mode=0o700)
        role_laws=[];inner_mass=None;inner_prepared=None
        for role in ("OSF_DEFENSE_FIT","ATTACK_FIT_NEW","INNER_SELECTION"):
            source=inner if role=="INNER_SELECTION" else load_allowed_role(manifest,seed,role)
            wire=source["wire"]
            if arm=="P0":
                a=wire["code_r1"][:,None];b=wire["code_r2"][:,None];mass=np.ones(a.shape,np.float64)
                p={"code_r1":a,"code_r2":b,"b0_code_r1":wire["code_r1"],"b0_code_r2":wire["code_r2"]}
            else:
                policy=policies[arm];p=policy.space.prepare(wire["code_r1"],wire["code_r2"])
                a=p["code_r1"];b=p["code_r2"];mass=policy.law_prepared(p)
            law_sha=write_npz(directory/(role+"_LAW.npz"),{"row_id":wire["row_id"],"unit":wire["unit"],"code_r1":a,"code_r2":b,"mass":mass})
            role_laws.append({"role":role,"law_sha256":law_sha,"rows":len(mass),"groups":len(np.unique(wire["unit"]))})
            if role=="INNER_SELECTION":inner_mass=mass;inner_prepared=p
        utility=expected_rows(inner_prepared,inner_mass,space,(inner["income"]["income"],inner["occupation"]["occupation"]),inner["references"])
        gates=[];rows={"row_id":inner["wire"]["row_id"],"unit":inner["wire"]["unit"]}
        summary={}
        for key,value in utility["expected"].items():summary[key]=float(value.mean())
        for ref,values in utility["excess"].items():
            for key,value in values.items():
                metric=key.rsplit("_",1)[1];cap=ORIGINAL_CAPS[metric];margin=float(value.mean())
                gates.append({"reference":ref,"task_metric":key,"margin":margin,"cap":cap,"pass":bool(margin<=cap)})
                rows[f"{ref}_{key}"]=value
        utility_sha=write_npz(directory/"INNER_UTILITY_ROWS.npz",rows)
        source_pin=None if arm=="P0" else manifest["seeds"][str(seed)]["policies"]["V_J" if arm=="HIST_V_J" else "V_S21"]["receipt"]["sha256"]
        receipt={"schema":"CBR1-frozen-baseline-v1","arm_id":arm,"source_seed":seed,"source_manifest_sha256":cfg["source_manifest_sha256"],
                 "historical_policy_receipt_sha256":source_pin,"original_bank_sha256":manifest["seeds"][str(seed)]["public_bank"]["sha256"],
                 "role_laws":role_laws,"inner_point_utility":gates,"inner_point_eligible":all(g["pass"] for g in gates),
                 "inner_expected_task_metrics":summary,"inner_utility_rows_sha256":utility_sha,"new_solver_calls":0,"new_teacher_fits":0,
                 "fit_labels_read":False,"sex_labels_read":False,"source_class_preserved":True}
        receipt_sha=write_once(directory/"POLICY_RECEIPT.json",receipt)
        arms.append({**receipt,"policy_receipt_sha256":receipt_sha})
    report={"schema":"CBR1-R3-baseline-restore-unit-v1","unit_id":cfg["unit_id"],"seed":seed,"status":"COMPLETE",
            "science_lock_sha256":cfg["science_lock_sha256"],"source_manifest_sha256":cfg["source_manifest_sha256"],
            "arms":arms,"new_solver_calls":0,"sex_labels_read":False,"closed_role_access":False,
            "role_law_forwards":9,"inner_task_labels_files_read":2,"inner_reference_files_read":1}
    write_once(cfg["public_output"],report,False)
    print(json.dumps({"status":"COMPLETE","unit_id":cfg["unit_id"],"arms":3,"point_eligible":sum(a["inner_point_eligible"] for a in arms)}))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args();run(args.config)

if __name__=="__main__":main()
