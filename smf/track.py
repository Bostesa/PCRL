"""Critic tracking at aligned frozen snapshots (inner roles only; pinned rgj.critic_track.compare).

Runs: the Phase A joint units at rho 0.75 for each schedule (schedule comparison at equal strength), and the Phase B
runs containing the selected (or descriptive) J-F and J-N of each seed. Snapshots: step 1, step 611 (midpoint) and the
last step (theta_{T-1} with the online critics aligned to it); refreshed runs also compare their diagnostic epoch-20
refit critics at theta_T. Registered statistic: mean over kinds of paired CE(online_j) - CE(fresh_j) on CONTROLLER_CALIB
(alias CALIB) and INNER_SELECTION rows; best-of-bank differences are kept separately.
"""
from __future__ import annotations

import json

import torch

from rgj import critic_track as CT
from rgj import finalize as FN
from rgj import train as RT
from smf import run as R


def jobs():
    out = [(k, f"A|{s}", R.a_run(k, "NJ", s, 0.75)) for k in R.SEEDS for s in R.SCHEDS]
    sb = R.RUN / "selection_B.json"
    if sb.exists():
        SB = json.loads(sb.read_text())
        for k in R.SEEDS:
            for arm in ("J-F", "J-N"):
                a = SB[str(k)]["arms"].get(arm)
                if a and a.get("rho") is not None:
                    out.append((k, f"B|{arm}", R.b_run(k, arm, a["rho"])))
    return out


def run_tracking(D, shard_spec=None):
    data = RT.TData(D)
    for k, lab, rn in R.shard(jobs(), shard_spec):
        name = f"track__s{k}__{lab.replace('|', '__')}"
        if R.done(name):
            continue
        cap = torch.load(R.U(rn) / "captures.pt")
        fin = torch.load(R.U(rn) / "final.pt")
        snaps = {"start": cap[1], "mid": cap[R.MID_STEP], "final": fin["theta_T_minus_1"]}
        res = {"seed": k, "label": lab, "run": rn, "snapshots": {}}
        for s, sn in snaps.items():
            res["snapshots"][s] = CT.compare(sn["model"], sn["critics"], sn["transforms"], sn["critic_head"], D, data, k, s)
        if "refit_critics" in fin:
            rc = fin["refit_critics"]
            res["final_refit_at_theta_T"] = CT.compare(fin["theta_T"], rc["critics"], rc["transforms"], rc["critic_head"],
                                                       D, data, k, "refitT")
        FN.save_unit(R.U(name), {}, res)
        R.event("unit complete", unit=name)
