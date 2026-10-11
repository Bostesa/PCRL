"""Queued ATT-only, label-free projection of actual positive HGB action rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from .attacks import attack_split,hgb_dense_projection
from .attack_worker import _queue_config,LAW_FIELDS
from .source import read_exact,write_once


def run(config):
    active,cfg,verify = _queue_config(config,"admission")
    results=[]
    for item in cfg["instances"]:
        for name in ("law","bank"):
            verify(item[name]["path"],item[name]["sha256"])
        law=read_exact(item["law"],LAW_FIELDS)
        bank=json.loads(Path(item["bank"]["path"]).read_text())
        split=attack_split(law["unit"],require_adult=True)
        mass=law["mass"][split["fit"]]
        positive=int(np.count_nonzero(mass>0))
        columns={}
        probability={}
        for r in (1,2):
            p=np.asarray(bank["recipients"][str(r)]["probability"])
            codes,classes=p.shape
            columns[f"r{r}"]=codes+max(1,(codes-1).bit_length())+classes*2+2
            probability[f"r{r}"]=classes
        columns["pair"]=sum(columns.values())
        probability["pair"]=sum(probability.values())
        projected={view:{"HGB_15x200":hgb_dense_projection(positive,columns[view]),
                         "PROB_HGB_15x200":hgb_dense_projection(positive,probability[view])}
                   for view in ("r1","r2","pair")}
        results.append({"arm_id":item["arm_id"],"seed":item["seed"],"fit_rows":len(split["fit"]),
            "actions":law["mass"].shape[1],"positive_fit_actions":positive,
            "law_sha256":item["law"]["sha256"],"bank_sha256":item["bank"]["sha256"],"projections":projected})
    result={"schema":"CBR1-label-free-reader-memory-preflight-v1","status":"COMPLETE","unit_id":active["unit_id"],
        "instances":results,"sex_access":False,"inner_access":False,"model_fits":0}
    write_once(cfg["public_output"],result,False)
    print(json.dumps({"status":"COMPLETE","instances":len(results),"sex_access":False}))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("--config",required=True)
    run(p.parse_args().config)


if __name__=="__main__":main()
