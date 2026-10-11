"""Queued public-receipt task arithmetic; no person arrays or labels are read."""
import argparse
import csv
import json
import math
from pathlib import Path

from .attack_worker import _queue_config
from .source import sha_file,write_once

CAPS={"accuracy":.010,"nll":.010,"brier":.005}


def point_records(receipt):
    values=[]
    if "arm_id" in receipt and "inner_point_utility" in receipt:values.append(receipt)
    for name in ("arms","setups"):
        values.extend(item for item in receipt.get(name,[]) if "arm_id" in item and "inner_point_utility" in item)
    return values


def point_margins(record):
    output={}
    for item in record["inner_point_utility"]:
        key=(item["reference"],item["task_metric"])
        if key in output or key[0] not in ("U0","Ucal") or key[1] not in {
                f"r{r}_{metric}" for r in (1,2) for metric in CAPS}:
            raise PermissionError("duplicate or unknown public task endpoint")
        output[key]=float(item["margin"])
    if len(output)!=12 or not all(math.isfinite(value) for value in output.values()):
        raise PermissionError("public point receipt lacks all twelve task/reference endpoints")
    return output


def absolute_cost(p0_cost,p0_margins,arm_margins,key):
    derived={ref:p0_cost[key]-p0_margins[(ref,key)]+arm_margins[(ref,key)] for ref in ("U0","Ucal")}
    difference=abs(derived["U0"]-derived["Ucal"])
    if difference>1e-12:
        raise PermissionError("U0/Ucal absolute task reconstruction disagrees")
    return derived["U0"],difference


def run(config):
    active,cfg,verify=_queue_config(config,"admission")
    if len(cfg["arm_ids"])!=17 or len(set(cfg["arm_ids"]))!=17:
        raise PermissionError("all seventeen registered arm IDs are required")
    records={};sources=[];p0={}
    for item in cfg["public_receipts"]:
        pin=item["pin"];path=verify(pin["path"],pin["sha256"])
        if "results" not in path.parts or ".private" in path.parts:
            raise PermissionError("arithmetic worker only admits public aggregate receipts")
        value=json.loads(path.read_text());sources.append(pin["sha256"])
        if value.get("status")!="COMPLETE":raise PermissionError("public input unit is incomplete")
        seed=item["seed"]
        if type(seed) is not int or seed not in (0,1):raise PermissionError("paired seed identifier differs")
        for record in point_records(value):
            identity=(record["arm_id"],seed)
            if identity in records:raise PermissionError("duplicate public arm/seed aggregate")
            records[identity]=point_margins(record)
            if record["arm_id"]=="P0":p0[seed]=record["inner_expected_task_metrics"]
    if set(p0)!={0,1}:raise PermissionError("both original P0 absolute task references are required")
    rows=[];maximum=0.;evaluated=0
    for arm in cfg["arm_ids"]:
        for seed in (0,1):
            margins=records.get((arm,seed));baseline=records[("P0",seed)]
            conjunction=None if margins is None else all(v<=CAPS[k.rsplit("_",1)[1]] for (_,k),v in margins.items())
            for recipient,task in (("r1","income"),("r2","occupation")):
                for metric,cap in CAPS.items():
                    key=f"{recipient}_{metric}"
                    expected=None;delta=None
                    if margins is not None:
                        expected,delta=absolute_cost(p0[seed],baseline,margins,key)
                        maximum=max(maximum,delta)
                    for ref in ("U0","Ucal"):
                        row={"arm_id":arm,"seed":seed,"recipient":recipient,"task":task,"reference":ref,"metric":metric,
                             "cap":cap,"status":"UNTESTED_DEPENDENCY" if margins is None else "COMPUTED_FROM_PUBLIC_RECEIPTS"}
                        if margins is not None:
                            reference=p0[seed][key]-baseline[(ref,key)]
                            row.update({"absolute_metric":-expected if metric=="accuracy" else expected,
                                "reference_absolute_metric":-reference if metric=="accuracy" else reference,
                                "mean_loss_excess":margins[(ref,key)],"accuracy_change":-margins[(ref,key)] if metric=="accuracy" else "",
                                "original_point_cap_pass":margins[(ref,key)]<=cap,"all_twelve_original_point_caps_pass":conjunction,
                                "dual_reference_absolute_difference":delta})
                            evaluated+=1
                        rows.append(row)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    out=Path(cfg["public_csv"])
    with out.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    receipt={"schema":"CBR1-public-absolute-task-aggregate-v1","status":"COMPLETE","unit_id":active["unit_id"],
        "rows":len(rows),"computed_rows":evaluated,"missing_rows":len(rows)-evaluated,"source_receipt_sha256":sources,
        "maximum_dual_reference_absolute_difference":maximum,"agreement_limit":1e-12,"public_csv_sha256":sha_file(out),
        "negative_accuracy_cost_in_source":True,"displayed_accuracy_is_positive":True,"person_arrays_read":False,
        "task_or_sensitive_labels_read":False,"model_fits":0,"calibration_fits":0,
        "scope":"fixed expected single-query law task means; no new scientific endpoints or uncertainty claims"}
    write_once(cfg["public_receipt"],receipt,False)
    print(json.dumps({"status":"COMPLETE","computed":evaluated,"missing":len(rows)-evaluated}))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("--config",required=True);run(p.parse_args().config)


if __name__=="__main__":main()
