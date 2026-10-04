"""Critic-tracking diagnosis at aligned frozen snapshots (PROTOCOL.md section 8; inner roles only).

For each seed and each of J-G, L-G, J-R, J-O, the run that contains the selected (or descriptive) checkpoint is read
from its saved snapshots:
  start  = Stage C step 1      (theta_0, critics after their step-1 updates, transform in use)
  mid    = Stage C step 761    (theta_760, ...)
  final  = last step 1520      (theta_{T-1}, the online critics after their last updates, transform in use)
At each snapshot, with the SAME frozen model, the SAME saved transform and the same rows, fresh critics of each kind are
trained (bounded refit on CRITIC_FIT, early stopping on CRITIC_VAL, deterministic init). Reported per view:
  registered gap = mean over kinds of CE(online_j) - CE(fresh_j)       on CALIB rows and on INNER_SELECTION rows
  best-of-bank   = min_j CE(online_j) - min_j CE(fresh_j)              (named separately)
plus the constant-prior CE on the same rows (so weak-vs-weak cannot pass as tracking). Refreshed arms also report their
epoch-20 refit critics against fresh critics at theta_T (the refit's own snapshot; diagnosis only).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from rgj import finalize as FN
from rgj import run as R
from rgj import train as T

SNAPS = ("start", "mid", "final")


def compare(model_state, critics, transforms, head, D, data, k, tag):
    model = R.model_from(model_state, k)
    Vdef = T.frozen_views(model, data.X, head)
    Vin = T.frozen_views(model, torch.from_numpy(np.asarray(D["X"][D["idx"]["INNER_SELECTION"]], dtype=np.float32)), head)
    Sin = torch.from_numpy(D["sex"][D["idx"]["INNER_SELECTION"]])
    cf, cv, cal = (torch.from_numpy(a) for a in (data.cf, data.cv, data.cal))
    lp = torch.tensor(np.log(data.prior), dtype=torch.float32)
    out = {}
    for v in T.VIEWS:
        Tv = T.Transform(state=transforms[v])
        Zf, Zv = Tv(Vdef[v][cf]), Tv(Vdef[v][cv])
        rows = {"calib": (Tv(Vdef[v][cal]), data.S[cal]), "inner": (Tv(Vin[v]), Sin)}
        rec = {"const_ce": {r: float(F.nll_loss(lp.expand(len(s), 2), s)) for r, (z, s) in rows.items()}, "kinds": {}}
        for kind in T.KINDS:
            on = T.critic(kind, T.DV[v])
            on.load_state_dict(critics[v][kind])
            fr = T.new_critic(k, v, kind, f"track-{tag}")
            fr, rc = T.fit_bounded(fr, Zf, data.S[cf], Zv, data.S[cv], [k, T._seed(v, kind), 777, T._seed(tag) % 997])
            rec["kinds"][kind] = {"online": {r: T.ce_of(on, z, s) for r, (z, s) in rows.items()},
                                  "fresh": {r: T.ce_of(fr, z, s) for r, (z, s) in rows.items()}, "fresh_fit": rc}
        for r in rows:
            on = [rec["kinds"][kk]["online"][r] for kk in T.KINDS]
            fr = [rec["kinds"][kk]["fresh"][r] for kk in T.KINDS]
            rec[f"gap_registered_{r}"] = float(np.mean([a - b for a, b in zip(on, fr)]))
            rec[f"gap_best_of_bank_{r}"] = float(min(on) - min(fr))
            rec[f"online_best_{r}"], rec[f"fresh_best_{r}"] = float(min(on)), float(min(fr))
        out[v] = rec
    return out


def selected_run(k, arm):
    sc = json.loads((R.RUN / "selection_C.json").read_text())[str(k)]["arms"][arm]
    return R.run_name("C", k, arm, sc["beta"]), sc["status"], sc["unit"]


BASELINE = "J-R_beta0"   # review A3: same L-R start, beta = 0 (critics train, encoder task-only): overfitting baseline


def baseline_run(D, data, k):
    st, crit, desc, _ = R.lr_init(k)
    _, diag, _, cap, fin = T.train_run("J-R", 0.0, st, data, k, "C", init_critics=crit,
                                       critic_head=T.head_of(R.load_warm(k)), capture_steps=(1, R.MID_STEP))
    return cap, fin, desc


def run_tracking(D, shard_spec=None):
    data = T.TData(D)
    jobs = [(k, a) for k in R.SEEDS for a in R.STAGE_C_ARMS + (BASELINE,)]
    for k, arm in R.shard(jobs, shard_spec):
        name = f"track__s{k}__{arm}"
        if R.done(name):
            continue
        if arm == BASELINE:
            cap, fin, unit = baseline_run(D, data, k)
            rn, status = "in-memory J-R beta=0 run from the frozen L-R (not a unit)", "BASELINE"
        else:
            rn, status, unit = selected_run(k, arm)
            cap = torch.load(R.U(rn) / "captures.pt")
            fin = torch.load(R.U(rn) / "final.pt")
        snaps = {"start": cap[1], "mid": cap[R.MID_STEP], "final": fin["theta_T_minus_1"]}
        res = {"seed": k, "arm": arm, "run": rn, "selected_unit": unit, "selection_status": status, "snapshots": {}}
        for s in SNAPS:
            sn = snaps[s]
            res["snapshots"][s] = compare(sn["model"], sn["critics"], sn["transforms"], sn["critic_head"], D, data, k, s)
        if "refit_critics" in fin:
            rc = fin["refit_critics"]
            res["final_refit_at_theta_T"] = compare(fin["theta_T"], rc["critics"], rc["transforms"], rc["critic_head"], D,
                                                    data, k, "refitT")
        FN.save_unit(R.U(name), {}, res)
        R.event("unit complete", unit=name)
