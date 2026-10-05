"""Critic tracking at aligned frozen snapshots (SELECTION_AND_AUDIT_LOCK; fitting/diagnostic roles only; pinned
rgj.critic_track.compare). Runs: the capture configurations (incumbent RAW-J beta 0.3, RAW-L beta 0.3, symmetric
NORM-J rho 3, NORM-L rho 3) on every seed. Snapshots at progress fractions 0, 0.25, 0.5, 0.75 (theta_{s-1} with the
critics after their step-s updates and the transform in use), the final aligned snapshot theta_{T-1}, and the released
theta_T read by the same final critics/transform (one encoder step stale). Registered statistic: mean over kinds of
CE(online critic) - CE(fresh bounded refit on the same frozen views, transform and rows) on DIAGNOSTIC_CALIB (alias
CALIB) and INNER_SELECTION rows; best-of-bank and constant-prior CE are reported separately. Diagnostics only: they never
change a model, a selection or the assessment attack bank.
"""
from __future__ import annotations

import torch

from osf import run as R
from osf import train as T
from rgj import critic_track as CT
from rgj import finalize as FN
from rgj import train as RT


def jobs():
    return [(k, cid) for k in R.SEEDS for cid in R.CAPTURE_CONFIGS]


def run_tracking(D, shard_spec=None):
    data = RT.TData(D)
    for k, cid in R.shard(jobs(), shard_spec):
        name = f"track__s{k}__{R.safe(cid)}"
        if R.done(name):
            continue
        rn = R.run_name(k, cid)
        cap = torch.load(R.U(rn) / "captures.pt")
        fin = torch.load(R.U(rn) / "final.pt")
        steps = T.capture_steps_for(40, data.n)
        snaps = {f"f{f:g}": cap[s] for f, s in steps.items() if f < 1.0}
        last = fin["theta_T_minus_1"]
        snaps["final_aligned_theta_T_minus_1"] = last
        res = {"seed": k, "config": cid, "run": rn, "steps": {str(f): s for f, s in steps.items()}, "snapshots": {}}
        for s, sn in snaps.items():
            res["snapshots"][s] = CT.compare(sn["model"], sn["critics"], sn["transforms"], sn["critic_head"], D, data,
                                             k, s)
        res["snapshots"]["released_theta_T_stale_critics"] = CT.compare(fin["theta_T"], last["critics"],
                                                                        last["transforms"], last["critic_head"], D,
                                                                        data, k, "thetaT")
        FN.save_unit(R.U(name), {}, res)
        R.event("unit complete", unit=name)
