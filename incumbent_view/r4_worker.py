"""Queue-only independent fresh attackers of exact P0-bank releases.

This module imports no protection trainer. Receipt/source/role/wire gates all
precede ATT SEX; INNER SEX remains lazy until the inherited reader freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import assert_same_people, sha_array
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import run_three_view_attack
from .r2_package import load_policy
from .r4_views import ARRAY_KEYS, package_from_arrays, prepare_three_views
from .r4_public_law import uniform_output_masses

ARMS=("P0","O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")
ROLES=("ATTACK_FIT_NEW","INNER_SELECTION")
LAW_KEYS=frozenset({"row_id","unit","code_r1","code_r2","mass","valid","b0_code_r1","b0_code_r2"})
CODE_KEYS=frozenset({"row_id","unit","code_r1","code_r2"})
SPEC_KEYS=frozenset({"schema","unit_id","arm_id","source_seed","realization","attacker_seed",
    "campaign_seal","role_manifest","public_bank","eligibility","source_pins","utility_links",
    "role_releases","sex_archives","policy_package","private_output_path","public_receipt_path"})
MODULES={
    "incumbent_view.r4_worker": "incumbent_view/r4_worker.py",
    "incumbent_view.r4_views": "incumbent_view/r4_views.py",
    "incumbent_view.r4_public_law": "incumbent_view/r4_public_law.py",
    "incumbent_view.r2_package": "incumbent_view/r2_package.py",
    "incumbent_view.r2_policy": "incumbent_view/r2_policy.py",
    "incumbent_view.r2_space": "incumbent_view/r2_space.py",
    "results.pcrl_incumbent_view_risk_v1.r0.compute_queue": "results/pcrl_incumbent_view_risk_v1/r0/compute_queue.py",
    "results.pcrl_latent_and_sampled_methods_v1.r01.role_store": "results/pcrl_latent_and_sampled_methods_v1/r01/role_store.py",
    "results.pcrl_latent_and_sampled_methods_v1.r02.protocol": "results/pcrl_latent_and_sampled_methods_v1/r02/protocol.py",
    "results.pcrl_latent_and_sampled_methods_v1.r02.complete_view": "results/pcrl_latent_and_sampled_methods_v1/r02/complete_view.py",
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire": "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/attacks/reference_wire.py",
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize": "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/reference_materialize.py",
    "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.wire": "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/attacks/wire.py",
    "attack_eval": "results/pcrl_method_portfolio_v1/shared/attack_eval.py",
    "attack_controls_v2": "results/pcrl_method_portfolio_v1/shared/attack_controls_v2.py",
    "common_eval": "results/pcrl_method_portfolio_v1/shared/common_eval.py",
    "safe_data_guard": "results/pcrl_method_portfolio_v1/shared/safe_data_guard.py",
}


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def validate_identity(spec):
    if (set(spec)!=SPEC_KEYS or spec.get("schema")!="pcrl-incumbent-r4-final-unit-v1" or spec.get("arm_id") not in ARMS or
            type(spec.get("source_seed")) is not int or spec["source_seed"] not in (0,1) or
            type(spec.get("realization")) is not int or spec["realization"] not in (0,1,2) or
            spec.get("attacker_seed") !=100000+1000*spec["source_seed"]+spec["realization"] or
            spec["arm_id"] in ("P0","V_D") and spec["realization"]!=0):
        raise PermissionError("fixed arm/seed/draw/reader identity differs")
    if set(spec.get("role_releases",{}))!=set(ROLES):
        raise PermissionError("exact ATT/INNER complete release required")


def validate_gate(gate,spec):
    if (gate.get("schema")!="pcrl-incumbent-r3-original-inner-point-gate-v1" or
            gate.get("arm_id")!=spec["arm_id"] or gate.get("seeds")!=[0,1] or
            gate.get("all_two_seed_task_reference_point_caps_pass") is not True or
            gate.get("original_inner")!={"required":24,"passed":24,"status":"PASS"}):
        raise PermissionError("fresh attacks require original INNER24-only point gate")
    if set(gate.get("source_receipts",{}))!={"s0_fit","s0_inner","s1_fit","s1_inner"}:
        raise PermissionError("point gate source-receipt inventory differs")


def _archive(path,keys):
    if ".private" not in path.parts or path.stat().st_mode&0o077:
        raise PermissionError("role release must have owner-only private custody")
    with ZipFile(path) as z:
        if len(z.infolist())!=len(keys) or {x.filename for x in z.infolist()}!={k+".npy" for k in keys}:
            raise PermissionError("role-only archive keys differ before deserialization")
    with np.load(path,allow_pickle=False) as a:return {k:np.array(a[k],copy=True) for k in keys}


def load_sex(pin,role,manifest,pinned):
    """Exact one-role one-label archive; authorization precedes NumPy reads."""
    if role not in ROLES or pin["sha256"]!=manifest["roles"][role]["source_files"]["sex"]["sha256"]:
        raise PermissionError("SEX archive role/hash is not admitted")
    record=_archive(pinned(pin),{"row_id","unit","sex"})
    meta=manifest["roles"][role]
    row,unit,y=record["row_id"],record["unit"],record["sex"]
    if (row.dtype!=np.int64 or row.shape!=(meta["rows"],) or unit.dtype!=np.int64 or unit.shape!=row.shape or
            not np.all(np.diff(row)>0) or len(np.unique(unit))!=meta["groups"] or
            sha_array(np.c_[row,unit])!=meta["row_unit_pair_sha256"] or
            y.dtype!=np.int64 or y.shape!=row.shape or np.any(y<0) or np.any(y>1)):
        raise PermissionError("SEX archive class/people schema differs")
    return record


def wire_hash(wires):
    digest=hashlib.sha256()
    for wire in wires:
        digest.update(json.dumps(wire,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n")
    return digest.hexdigest()


def check_law(law, arrays, bank):
    if not np.array_equal(law["row_id"],arrays["row_id"]) or not np.array_equal(law["unit"],arrays["unit"]):
        raise PermissionError("release/law person order differs")
    mass,valid=law["mass"],law["valid"]
    n=len(arrays["row_id"])
    if (mass.dtype!=np.float64 or mass.shape!=(n,16) or valid.dtype!=bool or valid.shape!=mass.shape or
            not np.isfinite(mass).all() or np.any(mass<0) or np.any(mass[~valid]!=0) or
            not np.allclose(mass.sum(1),1,atol=1e-12,rtol=0)):
        raise PermissionError("finite law mass/support invalid")
    found=np.ones(mass.shape,bool)
    for r in (1,2):
        key=f"code_r{r}";code=law[key];source=law[f"b0_code_r{r}"]
        if (code.dtype!=np.int64 or code.shape!=mass.shape or np.any(code<0) or
                np.any(code>=len(bank[f"bank_r{r}"])) or source.dtype!=np.int64 or source.shape!=(n,) or
                np.any(source<0) or np.any(source>=len(bank[f"bank_r{r}"])) or
                arrays[key].dtype!=np.int64 or arrays[key].shape!=(n,) or
                np.any(arrays[key]<0) or np.any(arrays[key]>=len(bank[f"bank_r{r}"])) or
                not np.array_equal(code[:,0],source)):
            raise PermissionError("global action/source code schema differs")
        decision=bank[f"bank_r{r}"].argmax(1)
        if np.any((decision[code]!=decision[source,None]) & valid):
            raise PermissionError("selected source support changes P0 disclosed class")
        found &= code==arrays[key][:,None]
    if not np.all(np.any(found & valid & (mass>0),axis=1)):
        raise PermissionError("cached release outside positive joint-law support")


def run(spec_path,spec_sha256,stage_lock_path,stage_lock_sha256):
    active=assert_active_from_env(stage="attacks")
    state=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered={Path(x["path"]).resolve():x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}
    def pinned(pin):
        path=Path(pin["path"])
        if path.is_symlink():raise PermissionError("pinned artifact cannot be symlink")
        path=path.resolve(strict=True)
        if registered.get(path)!=pin["sha256"] or sha(path)!=pin["sha256"]:
            raise PermissionError("artifact is not the active queue's exact pinned bytes")
        return path
    def read(pin):return json.loads(pinned(pin).read_text())
    spec=read({"path":spec_path,"sha256":spec_sha256});validate_identity(spec)
    if spec["unit_id"]!=active["unit_id"] or spec["campaign_seal"]["sha256"]!=active["seal_sha256"]:
        raise PermissionError("worker campaign/unit differs from active queue")
    seal=pinned(spec["campaign_seal"])
    lock=read({"path":stage_lock_path,"sha256":stage_lock_sha256})
    expected={"spec_sha256":spec_sha256,"arm_id":spec["arm_id"],"source_seed":spec["source_seed"],
              "realization":spec["realization"],"point_eligible":True}
    if (lock.get("schema")!="pcrl-incumbent-final-attack-stage-lock-v1" or
            lock.get("status")!="LOCALLY_SEALED_ATTACKS" or lock.get("units",{}).get(spec["unit_id"])!=expected or
            state["units"][spec["unit_id"]]["spec"].get("amendment_sha256")!=stage_lock_sha256 or
            not any(a["sha256"]==stage_lock_sha256 and "attacks" in a["stages"] for a in state["amendments"])):
        raise PermissionError("exact attack stage lock/amendment not admitted")
    pins={item["module"]:item for item in spec["source_pins"]}
    if set(pins)!=set(MODULES):raise PermissionError("import closure differs")
    for module in MODULES:
        imported=Path(importlib.import_module(module).__file__).resolve()
        if pinned(pins[module])!=imported:raise PermissionError("hash-pinned source is not live imported module")
    gate=read(spec["eligibility"]);validate_gate(gate,spec)
    if set(spec["utility_links"])!=set(gate["source_receipts"]):raise PermissionError("utility link inventory differs")
    scored={}
    for key,item in spec["utility_links"].items():
        receipt=read(item["receipt"]);pinned(item["csv"])
        source=gate["source_receipts"][key]
        if source!={"receipt_sha256":item["receipt"]["sha256"],"csv_sha256":item["csv"]["sha256"],
                    "unit_id":receipt["unit_id"]}:
            raise PermissionError("gate/scorer receipt link differs")
        role="OSF_DEFENSE_FIT" if key.endswith("fit") else "INNER_SELECTION"
        if (receipt["schema"]!="pcrl-incumbent-r3-utility-receipt-v1" or
                receipt["arm_id"]!=spec["arm_id"] or receipt["source_seed"]!=int(key[1]) or
                receipt["role"]!=role or receipt["csv_sha256"]!=item["csv"]["sha256"]):
            raise PermissionError("gate scorer arm/seed/role/CSV differs")
        if spec["arm_id"]!="P0":
            law_receipt=read(item["law_receipt"]);policy=read(item["policy_receipt"])
            if (receipt["law_file_sha256"]!=item["law"]["sha256"] or
                    receipt["policy_receipt_sha256"]!=item["policy_receipt"]["sha256"] or
                    law_receipt["policy_receipt_sha256"]!=item["policy_receipt"]["sha256"] or
                    law_receipt["law_file_sha256"]!=item["law"]["sha256"] or
                    law_receipt["law_content_sha256"]!=receipt["law_content_sha256"] or
                    policy["arm_id"]!=spec["arm_id"] or policy["source_seed"]!=int(key[1]) or
                    receipt["source_unit_id"]!=policy["source_unit_id"] or
                    law_receipt["source_unit_id"]!=policy["source_unit_id"]):
                raise PermissionError("utility law/policy provenance differs")
            pinned(item["law"])
        scored[key]=receipt
    public_bank=read(spec["public_bank"])
    if public_bank["schema"]!="pcrl-incumbent-original-public-bank-v1" or public_bank["seed"]!=spec["source_seed"]:
        raise PermissionError("original decoder bank identity differs")
    identity={"release_id":public_bank["release_id"],"source_binding_sha256":public_bank["source_binding_sha256"]}
    for r in (1,2):
        b=public_bank["recipients"][str(r)];p=np.asarray(b["probability"],np.float64)
        if b["global_ids"]!=list(range(len(p))) or b["bank_sha256"]!=sha_array(p) or b["code_length_bits"]!=(len(p)-1).bit_length():
            raise PermissionError("public original ID/table/metadata differs")
        identity[f"bank_r{r}"]=p
    manifest=read(spec["role_manifest"])
    if manifest.get("schema")!="pcrl-incumbent-r1-data-access-manifest-v1" or manifest.get("status")!="EXACT_ROLE_ONLY_SOURCES_ADMITTED":
        raise PermissionError("new role-only data admission absent")
    for key,receipt in scored.items():
        if receipt["source_binding_sha256"]!=manifest["seeds"][key[1]]["source_binding_sha256"]:
            raise PermissionError("utility scorer has different original source bank")
        if spec["arm_id"]=="P0" and (receipt["policy_receipt_sha256"] is not None or
                receipt["source_unit_id"]!=f"R1_P0_S{key[1]}_ADMISSION"):
            raise PermissionError("P0 utility is not original admitted identity source")
    source_identity=manifest["seeds"][str(spec["source_seed"])]
    if (source_identity["public_bank_sha256"]!=spec["public_bank"]["sha256"] or
            source_identity["source_binding_sha256"]!=identity["source_binding_sha256"] or
            source_identity["source_release_id"]!=identity["release_id"]):
        raise PermissionError("source bank/metadata differs from new R1 admission")
    if set(spec["sex_archives"])!=set(ROLES):raise PermissionError("exact two authorized SEX pins required")
    if spec["arm_id"]=="P0":
        if spec["policy_package"] is not None:raise PermissionError("P0 cannot have a refitted policy")
        policy=None
    else:
        policy_record=read(spec["policy_package"]["receipt"])
        package_path=pinned(spec["policy_package"]["archive"])
        if (policy_record.get("schema")!="IVR1-selected-policy-v1" or
                policy_record["arm_id"]!=spec["arm_id"] or
                policy_record["source_seed"]!=spec["source_seed"] or
                policy_record["source_binding_sha256"]!=identity["source_binding_sha256"] or
                policy_record["public_bank_sha256"]!=spec["public_bank"]["sha256"] or
                policy_record["policy_package_sha256"]!=spec["policy_package"]["archive"]["sha256"] or
                package_path.parent!=(pinned(spec["policy_package"]["receipt"])).parent or
                policy_record["source_unit_id"]!=package_path.parent.parent.name or
                scored[f"s{spec['source_seed']}_inner"]["policy_receipt_sha256"]!=spec["policy_package"]["receipt"]["sha256"]):
            raise PermissionError("selected published policy/deployment package differs")
        policy=load_policy(package_path.parent,expected_receipt_sha256=spec["policy_package"]["receipt"]["sha256"])
        for r in (1,2):
            if not np.array_equal(getattr(policy.space,f"probability{r}"),identity[f"bank_r{r}"]):
                raise PermissionError("selected policy changes P0 decoder/order")
    packages={}
    invocation_constants={}
    for role,item in spec["role_releases"].items():
        receipt=read(item["receipt"])
        arrays=_archive(pinned(item["codes"]),ARRAY_KEYS if spec["arm_id"]=="P0" else CODE_KEYS)
        meta=manifest["roles"][role]
        if (len(arrays["row_id"])!=meta["rows"] or len(np.unique(arrays["unit"]))!=meta["groups"] or
                sha_array(np.c_[arrays["row_id"],arrays["unit"]])!=meta["row_unit_pair_sha256"] or
                receipt["row_unit_pair_sha256"]!=meta["row_unit_pair_sha256"]):
            raise PermissionError("release role/person binding differs")
        if spec["arm_id"]=="P0":
            admitted=source_identity["role_releases"][role]
            if (receipt["release_sha256"]!=item["codes"]["sha256"] or
                    admitted["sha256"]!=item["codes"]["sha256"] or
                    admitted["receipt_sha256"]!=item["receipt"]["sha256"]):
                raise PermissionError("exact P0 release file differs")
        else:
            if (receipt["arm_id"]!=spec["arm_id"] or receipt["source_seed"]!=spec["source_seed"] or
                    receipt["role"]!=role or receipt["realization"]!=spec["realization"] or
                    receipt["source_binding_sha256"]!=identity["source_binding_sha256"] or
                    receipt["codes_file_sha256"]!=item["codes"]["sha256"] or
                    receipt["law_file_sha256"]!=item["law"]["sha256"] or
                    receipt["source_unit_id"]!=policy_record["source_unit_id"]):
                raise PermissionError("selected role/draw/source receipt differs")
            law=_archive(pinned(item["law"]),LAW_KEYS);check_law(law,arrays,identity)
            utility=scored[f"s{spec['source_seed']}_inner"]
            if receipt["policy_receipt_sha256"]!=utility["policy_receipt_sha256"]:
                raise PermissionError("attacked policy differs from eligible utility policy")
            if role=="INNER_SELECTION":
                admitted_law=_archive(pinned(spec["utility_links"][f"s{spec['source_seed']}_inner"]["law"]),LAW_KEYS)
                if any(not np.array_equal(law[k],admitted_law[k]) for k in LAW_KEYS):
                    raise PermissionError("cached draw law differs from original utility law")
            for r in (1,2):
                probability=identity[f"bank_r{r}"][arrays[f"code_r{r}"]]
                arrays[f"prob_r{r}"]=probability
                arrays[f"decision_r{r}"]=probability.argmax(1).astype(np.int64)
        packages[role]=package_from_arrays(arrays,identity=identity,role=role)
        actual_wire={"r1":wire_hash(packages[role].wires1),"r2":wire_hash(packages[role].wires2)}
        if receipt.get("wire_sha256_by_recipient")!=actual_wire:
            raise PermissionError("attack wire differs from metadata-bearing deployed role wire")
        invocation_constants[role]={"seed":spec["source_seed"],"arm_id":spec["arm_id"],
            "policy_receipt_sha256":None if spec["policy_package"] is None else spec["policy_package"]["receipt"]["sha256"],
            "rows":len(arrays["row_id"]),"draw_kind":receipt.get("draw_kind","none"),
            "recipient_jsonl_release_sha256":actual_wire}
    masses=uniform_output_masses(identity["bank_r1"],identity["bank_r2"],policy)
    prepared=prepare_three_views(packages[ROLES[0]],packages[ROLES[1]],uniform_masses=masses)
    output=Path(spec["private_output_path"])
    if output.exists() or ".private" not in output.parts or output.parent.stat().st_mode&0o077:
        raise PermissionError("new owner-only output parent required")
    if Path(spec["public_receipt_path"]).exists():raise FileExistsError("public attack receipt exists")
    def guard():
        current=assert_active_from_env(unit_id=spec["unit_id"],stage="attacks")
        if current["seal_sha256"]!=spec["campaign_seal"]["sha256"]:raise PermissionError("campaign seal changed")
    guard()
    sex_att=load_sex(spec["sex_archives"]["ATTACK_FIT_NEW"],"ATTACK_FIT_NEW",manifest,pinned)
    assert_same_people(sex_att,{"row_id":packages[ROLES[0]].row_id,"unit":packages[ROLES[0]].unit})
    def inner_sex():
        guard()
        if not (output/"FROZEN_COMPLETE.json").is_file():raise PermissionError("INNER labels embargo until reader freeze")
        record=load_sex(spec["sex_archives"]["INNER_SELECTION"],"INNER_SELECTION",manifest,pinned)
        assert_same_people(record,{"row_id":packages[ROLES[1]].row_id,"unit":packages[ROLES[1]].unit})
        return record
    attack=run_three_view_attack(prepared,sex_att,load_inner_sex=inner_sex,family="A",
        seed=spec["attacker_seed"],private_root=output,lock_path=seal,
        lock_sha256=spec["campaign_seal"]["sha256"],queue_guard=guard)
    public={"schema":"pcrl-incumbent-r4-fresh-attack-v1","status":"COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
            "unit_id":spec["unit_id"],"arm_id":spec["arm_id"],"source_seed":spec["source_seed"],
            "realization":spec["realization"],"attacker_seed":spec["attacker_seed"],
            "campaign_seal_sha256":spec["campaign_seal"]["sha256"],"stage_lock_sha256":stage_lock_sha256,
            "unit_spec_sha256":spec_sha256,"public_bank_sha256":spec["public_bank"]["sha256"],
            "eligibility_sha256":spec["eligibility"]["sha256"],
            "public_law_feature_contract":{
                "source_prior":masses["source_prior"],
                "derived_column_order":{
                    "r1":["public_log_probability","public_uniform_source_log_output_mass"],
                    "r2":["public_log_probability","public_uniform_source_log_output_mass"],
                    "pair":["r1.public_log_probability","r1.public_uniform_source_log_output_mass",
                            "r2.public_log_probability","r2.public_uniform_source_log_output_mass",
                            "pair_code_bit_products","pair_public_uniform_source_log_output_mass"]},
                "table_sha256":{"r1":sha_array(masses["r1"]),"r2":sha_array(masses["r2"]),
                                "pair":sha_array(masses["pair"])},
                "checkpoint_receipt_sha256":None if spec["policy_package"] is None else spec["policy_package"]["receipt"]["sha256"]},
            "role_release_sha256":{role:item["receipt"]["sha256"] for role,item in spec["role_releases"].items()},
            "public_invocation_constants_by_role":invocation_constants,
            "attack":attack["public"]}
    with os.fdopen(os.open(spec["public_receipt_path"],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),"w") as f:
        json.dump(public,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return {"unit_id":spec["unit_id"],"status":public["status"],"receipt_sha256":sha(spec["public_receipt_path"])}


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for name in ("spec","spec-sha256","stage-lock","stage-lock-sha256"):parser.add_argument("--"+name,required=True)
    a=parser.parse_args();print(json.dumps(run(a.spec,a.spec_sha256,a.stage_lock,a.stage_lock_sha256),sort_keys=True))


if __name__=="__main__":main()
