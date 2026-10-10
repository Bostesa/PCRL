"""Receipt-only accounting for the fixed final-attack slots.

This reads public attack receipts and hash-pinned worker specs. It never opens
labels, predictions or fitted readers, and makes no interval or winner claim.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from pathlib import Path
from .r4_worker import ARMS

VIEWS=("r1","r2","pair")
STOCHASTIC=("O_J","V_J","V_P","V_L","V_S12","V_S21")


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def pinned_relative(repo,pin):
    if set(pin)!={"path","sha256"}:raise PermissionError("exact relative source pin required")
    relative=Path(pin["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("public source path cannot expose home or escape repository")
    path=repo/relative
    if path.is_symlink() or not path.is_file() or sha(path)!=pin["sha256"]:
        raise PermissionError("source receipt changed")
    return path


def fixed_slots():
    return [dict(unit_id=f"R4_FINAL_{arm}_S{seed}_R{draw}",arm_id=arm,
                 source_seed=seed,realization=draw)
            for arm in ARMS for seed in (0,1)
            for draw in (range(3) if arm in STOCHASTIC else range(1))]


def _finite_auc(value,*,envelope):
    if (type(value) not in (float,int) or not math.isfinite(value) or
            value<(.5 if envelope else 0.) or value>1):
        raise PermissionError("fresh recovery AUC outside finite bounds")
    return float(value)


def aggregate(repo,prepared_pin,effective_pin):
    repo=Path(repo).resolve(strict=True)
    prepared=json.loads(pinned_relative(repo,prepared_pin).read_text())
    effective=json.loads(pinned_relative(repo,effective_pin).read_text())
    slots=fixed_slots();ids={x["unit_id"] for x in slots}
    actual_slots=prepared.get("all_slots",[])
    if (prepared.get("schema")!="pcrl-incumbent-r4-prepared-index-v1" or
            len(actual_slots)!=40 or
            any({k:s.get(k) for k in x}!=x or
                s.get("status") not in ("ATTACK_PLANNED","POINT_INELIGIBLE","INCOMPLETE_SEARCH")
                for x,s in zip(slots,actual_slots)) or
            effective.get("schema")!="pcrl-incumbent-r4-effective-attack-map-v1" or
            set(effective.get("slots",{}))!=ids):
        raise PermissionError("fixed forty-slot inventory differs")
    rows=[];units=[];complete=0;control_pass=0
    for slot in prepared["all_slots"]:
        uid=slot["unit_id"];source=effective["slots"][uid]
        if source is None:
            state="INCOMPLETE_ATTACK" if slot["status"]=="ATTACK_PLANNED" else slot["status"]
            units.append({**slot,"effective_status":state,"receipt_sha256":None,
                          "all_control_gates_pass":None})
            for view in VIEWS:
                rows.append({**slot,"view":view,"effective_status":state,"effective_unit_id":"",
                             "primary_envelope_auc":"","validation_winner_inner_auc":"",
                             "control_status":"","receipt_sha256":"","view_schema_sha256":""})
            continue
        if slot["status"]!="ATTACK_PLANNED" or set(source)!={"receipt","worker_spec","stage_lock"}:
            raise PermissionError("noneligible or unpinned effective attack result")
        receipt=json.loads(pinned_relative(repo,source["receipt"]).read_text())
        spec=json.loads(pinned_relative(repo,source["worker_spec"]).read_text())
        lock=json.loads(pinned_relative(repo,source["stage_lock"]).read_text())
        if (receipt.get("schema")!="pcrl-incumbent-r4-fresh-attack-v1" or
                receipt.get("status")!="COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
                any(receipt.get(k)!=slot[k] for k in ("arm_id","source_seed","realization")) or
                any(spec.get(k)!=slot[k] for k in ("arm_id","source_seed","realization")) or
                receipt["unit_id"]!=spec["unit_id"] or
                receipt["unit_spec_sha256"]!=source["worker_spec"]["sha256"] or
                receipt["stage_lock_sha256"]!=source["stage_lock"]["sha256"] or
                lock.get("units",{}).get(spec["unit_id"],{}).get("spec_sha256")!=source["worker_spec"]["sha256"] or
                receipt["eligibility_sha256"]!=spec["eligibility"]["sha256"] or
                receipt["public_bank_sha256"]!=spec["public_bank"]["sha256"] or
                receipt["queue_seal_sha256"]!=spec["queue_seal"]["sha256"] or
                receipt["campaign_seal_sha256"]!=spec["campaign_seal"]["sha256"] or
                receipt["role_release_sha256"]!={role:item["receipt"]["sha256"]
                                                     for role,item in spec["role_releases"].items()}):
            raise PermissionError("effective receipt differs from exact release/lock")
        attack=receipt["attack"]
        if (attack.get("schema")!="pcrl-next-frozen-attack-unit-v1" or attack.get("family")!="A" or
                attack.get("model_seed")!=spec["attacker_seed"] or
                attack["plan"]["fit_forecast"]["mandatory_fits"]!=30 or
                set(attack["scores"])!=set(VIEWS) or set(attack["controls"])!=set(VIEWS) or
                set(attack["reader_custody"])!=set(VIEWS) or
                set(attack["view_schema_sha256"])!=set(VIEWS)):
            raise PermissionError("fresh reader slate/view coverage differs")
        all_pass=True
        for view in VIEWS:
            score=attack["scores"][view];control=attack["controls"][view]["scores"]
            auc=_finite_auc(score["primary_envelope_auc"],envelope=True)
            winner=_finite_auc(score["validation_winner_inner_auc"],envelope=False)
            gates=control.get("gates",{})
            if (control.get("status")!="PASS" or
                    set(gates)!={"null_exact","planted_at_least_0.99","shuffled_independent_inner_at_most_0.60"} or
                    not all(value is True for value in gates.values())):all_pass=False
            rows.append({**slot,"view":view,"effective_status":"COMPLETE",
                         "effective_unit_id":spec["unit_id"],"primary_envelope_auc":auc,
                         "validation_winner_inner_auc":winner,"control_status":control.get("status"),
                         "receipt_sha256":source["receipt"]["sha256"],
                         "view_schema_sha256":attack["view_schema_sha256"][view]})
        complete+=1;control_pass+=int(all_pass)
        units.append({**slot,"effective_status":"COMPLETE","effective_unit_id":spec["unit_id"],
                      "receipt_sha256":source["receipt"]["sha256"],
                      "stage_lock_sha256":source["stage_lock"]["sha256"],
                      "all_control_gates_pass":all_pass})
    return {"schema":"pcrl-incumbent-r4-fixed-attack-aggregate-v1",
            "scope":"reused INNER development; point AUC accounting only, no intervals or claim decision",
            "prepared_index_sha256":prepared_pin["sha256"],"effective_map_sha256":effective_pin["sha256"],
            "fixed_slots":40,"completed_instances":complete,"incomplete_or_ineligible_instances":40-complete,
            "completed_views":3*complete,"expected_reader_and_control_fits":30*complete,
            "completed_instances_with_all_controls_pass":control_pass,"units":units},rows


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for field in ("repo","prepared-index","prepared-sha256","effective-map","effective-sha256","output-json","output-csv"):
        p.add_argument("--"+field,required=True)
    a=p.parse_args();repo=Path(a.repo).resolve(strict=True)
    def relative(path):return str(Path(path).resolve(strict=True).relative_to(repo))
    result,rows=aggregate(repo,{"path":relative(a.prepared_index),"sha256":a.prepared_sha256},
                          {"path":relative(a.effective_map),"sha256":a.effective_sha256})
    if len(rows)!=120:raise PermissionError("fixed complete-view row inventory differs")
    for path in (a.output_json,a.output_csv):
        if Path(path).exists():raise FileExistsError("aggregate output is write-once")
    with os.fdopen(os.open(a.output_csv,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result["auc_csv_sha256"]=sha(a.output_csv)
    with os.fdopen(os.open(a.output_json,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),"w") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"aggregate_sha256":sha(a.output_json),"completed":result["completed_instances"]},sort_keys=True))


if __name__=="__main__":main()
