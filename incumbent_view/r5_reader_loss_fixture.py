"""Tiny locked-form checks for independent frozen-reader loss replay."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_reader_loss_replay import fold_of, independent_loss_for_reader, public_feature


def run(output):
    active=assert_active_from_env(stage="verification")
    bank={"recipients":{"1":{"probability":[[.8,.2],[.3,.7]]},
                        "2":{"probability":[[.7,.3],[.2,.8]]}}}
    # Find one synthetic group per fold without using any study row identity.
    groups=[]
    for k in range(3):
        groups.append(next(v for v in range(100) if int(fold_of([v])[0])==k))
    fold=fold_of(groups)
    sex=np.asarray([0,1,0],np.int64)
    ids=np.zeros((3,16),np.int64)
    view_cache={"r1":(ids,2,bank)}
    checks=[]

    prior_arrays={}
    prior_loss=np.empty((3,16))
    count_arrays={}
    cat_loss=np.empty((3,16))
    nonlinear_arrays={}
    nonlinear_loss=np.full((3,16),math.log(2.))
    feature_width=public_feature(bank,"r1",np.arange(2)).shape[1]
    for k in range(3):
        train=fold!=k
        prior=(int(sex[train].sum())+1)/(int(train.sum())+2)
        prior_arrays[f"r0_f{k}"]=np.asarray([prior])
        prior_loss[k]=(-math.log(prior) if sex[k] else -math.log1p(-prior))
        count=np.zeros((2,2),np.float64)
        count[0,0]=int((sex[train]==0).sum())
        count[0,1]=int((sex[train]==1).sum())
        count_arrays[f"r1_f{k}"]=count
        cat=(count[0,1]+1)/(count[0].sum()+2)
        cat_loss[k]=(-math.log(cat) if sex[k] else -math.log1p(-cat))
        prefix=f"r2_f{k}"
        nonlinear_arrays.update({prefix+"_mean":np.zeros(feature_width),
                                 prefix+"_scale":np.ones(feature_width),
                                 prefix+"_w0":np.zeros((feature_width,64)),
                                 prefix+"_w1":np.zeros(64),
                                 prefix+"_w2":np.zeros((64,32)),
                                 prefix+"_w3":np.zeros(32),
                                 prefix+"_w4":np.zeros((32,1)),
                                 prefix+"_w5":np.zeros(1)})
    for i,family,arrays,loss in ((0,"prior",prior_arrays,prior_loss),
                                 (1,"categorical",count_arrays,cat_loss),
                                 (2,"nonlinear",nonlinear_arrays,nonlinear_loss)):
        arrays[f"r{i}_loss"]=loss
        gap,_=independent_loss_for_reader(i,{"view":"r1","name":f"s0r0/r1/{family}"},
                                          arrays,sex,fold,view_cache)
        if gap>1e-14:raise AssertionError("finite frozen-reader fixture differs")
        checks.append(family)
    result={"schema":"IVR1-r5-reader-loss-datafree-fixture-v1","status":"PASS",
            "unit_id":active["unit_id"],"checked_families":checks,
            "synthetic_rows":3,"real_rows_read":0,"model_fits":0}
    out=Path(output)
    with os.fdopen(os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644),"w") as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    print(json.dumps(run(p.parse_args().output),sort_keys=True))


if __name__=="__main__":main()
