"""Orchestrates stages 4-6 for one dataset (one worker per dataset; at most two workers).

    python -m oar.run --dataset adult --lock results/combined_output_aware_removal_v1/EXECUTION_LOCK.json

Refuses unless the execution lock verifies. Fixed order: seeds 0,1,2; within a seed: references, output surfaces,
untreated / LEACE / noise controls, FARE grid -> validation-only nominee -> views, zero-fairness control, utility,
certificates, real-data controls (seed 0 only, one per new interface type). Resumes by skipping hash-complete units.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import study as S


def ledger_add(ds, uid, cpu, fits):
    S.RUN.mkdir(parents=True, exist_ok=True)
    with open(S.RUN / f"LEDGER_{ds}.jsonl", "a") as f:
        f.write(json.dumps({"uid": uid, "cpu_s": cpu, "fits": fits, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime())}) + "\n")


def spent_cpu_s() -> float:
    tot = 0.0
    for p in S.RUN.glob("LEDGER_*.jsonl"):
        for line in p.read_text().splitlines():
            tot += json.loads(line)["cpu_s"]
    return tot


def run_dataset(ds: str, lock: dict, log=print):
    E, auth, syn = S.effective(), S.auth_real(), False
    c = S.CELLS[ds]
    W = S.load_world(ds)
    s, t, K_s, K_t = W["s"], W["t"], c["K_s"], c["K_t"]
    budget = float(lock["runtime"]["cpu_hours_ceiling"]) * 3600 - float(lock["runtime"]["reserve_cpu_s"])
    fare = lock["fare"]
    led = lambda uid, cpu, n: ledger_add(ds, uid, cpu, n)  # noqa: E731
    status = {"dataset": ds, "units": [], "stopped": None}

    def guard(uid):
        if spent_cpu_s() > budget:
            status["stopped"] = {"before": uid, "spent_cpu_s": spent_cpu_s(), "budget_cpu_s": budget}
            raise StopIteration(uid)

    def atk(uid, X, finite=False, plus=None, contract=None):
        guard(uid)
        r = S.attack_unit(uid, X, W, s, K_s, E, auth, syn, finite=finite, plus=plus, contract=contract, cpu_ledger=led)
        status["units"].append(uid)
        log(f"[{ds}] {uid} done")
        return r

    def u2(uid, X):
        guard(uid)
        r = S.u2_unit(uid, X, W, t, K_t, E, auth, syn, cpu_ledger=led)
        status["units"].append(uid)
        return r

    try:
        for k in S.SEEDS:
            P = f"{ds}__s{k}"
            D = S.load_seed(ds, k)
            assert np.array_equal(D["row_id"], W["row_id"])
            H, O = D["H"], D["O"]
            S.reference_unit(f"{P}__REF", W, s, t, K_s, K_t, E, auth, syn)
            # ---- stage 4: output surfaces (outputs-only recipient)
            var, hard = S.output_variants(O)
            (S.RUN / "aliases").mkdir(parents=True, exist_ok=True)
            (S.RUN / "aliases" / f"{P}.json").write_text(json.dumps(S.alias_report(O), indent=1))
            for v in ("full", "prob", "hard"):
                atk(f"{P}__O_{v}", var[v], finite=(v == "hard"), contract={"outputs": v, "features": None})
            # ---- controls: untreated, target LEACE, policy LEACE
            arms = {"A": H}
            arms["B"], _ = S.leace_release(ds, k, "B", H)
            arms["C"], _ = S.leace_release(ds, k, "C", H)
            heads = {}
            for a, X in arms.items():
                atk(f"{P}__{a}__rep", X, contract={"features": a, "outputs": None})
                atk(f"{P}__{a}__rep+clean", np.hstack([X, O]),
                    plus={"rep_unit": f"{P}__{a}__rep", "out_unit": f"{P}__O_full"},
                    contract={"features": a, "outputs": "historical clean logits"})
                u2(f"{P}__U2__{a}", X)
                if a in ("A", "B"):
                    heads[a] = S.head_unit(f"{P}__HEAD__{a}", X, W, t, K_t, E, auth, syn, cpu_ledger=led)
                    Oh = heads[a]["outputs"]
                    atk(f"{P}__O_head{a}", Oh, contract={"outputs": f"head on {a} features", "features": None})
                    atk(f"{P}__{a}__rep+head", np.hstack([X, Oh]),
                        plus={"rep_unit": f"{P}__{a}__rep", "out_unit": f"{P}__O_head{a}"},
                        contract={"features": a, "outputs": f"head fitted on {a} features only"})
            # ---- noise at the frozen sigma* (validation-selected in the benchmark), three release draws
            for rs in S.RELEASE_SEEDS:
                R = S.noise_release(W, H, S.SIGMA_STAR[ds], rs)
                q = f"{P}__D_rs{rs}"
                atk(f"{q}__rep", R, contract={"features": f"noise sigma={S.SIGMA_STAR[ds]} rs={rs}", "outputs": None})
                for v in ("full", "prob", "hard"):
                    atk(f"{q}__rep+{v}", np.hstack([R, var[v]]),
                        plus={"rep_unit": f"{q}__rep", "out_unit": f"{P}__O_{v}"},
                        contract={"features": "noise", "outputs": v})
                u2(f"{P}__U2__D_rs{rs}", R)
            # ---- FARE: frozen grid, validation-only nominee
            from . import fare_run as FR
            sel = FR.run_fare_seed(ds, k, P, W, H, O, var, arms, heads, fare, E, auth, syn, atk, u2, led, log)
            if k == S.SEEDS[0]:
                ctl_inputs = {"O_prob": (var["prob"], False), "O_hard": (var["hard"], True),
                              "A__rep+head": (np.hstack([H, heads["A"]["outputs"]]), False),
                              "D_rs0__rep+hard": (np.hstack([S.noise_release(W, H, S.SIGMA_STAR[ds], 0), var["hard"]]), False)}
                from . import fare_official as FO
                jsrc = sel["nominee_unit_source"]
                cells = np.load(S.unit_dir(f"{P}__FAREFIT_c{jsrc}") / "cells.npy")
                XF = S.onehot(cells, int(cells.max()) + 1)
                hF = np.load(S.unit_dir(f"{P}__HEAD__F") / "preds.npz")["head_outputs_all"]
                ctl_inputs.update({"F__rep": (XF, True), "F__rep+clean": (np.hstack([XF, O]), False),
                                   "F__rep+head": (np.hstack([XF, hF]), False)})
                ctl = {}
                for name, (X, fin) in ctl_inputs.items():
                    guard(f"{P}__CTL__{name}")
                    import time as _t
                    t0 = _t.process_time()
                    ctl[name] = S.null_and_planted(f"{P}__CTL__{name}", X, W, s, K_s, E, auth, syn, finite=fin)
                    led(f"{P}__CTL__{name}", _t.process_time() - t0, 0)
                (S.RUN / "controls").mkdir(parents=True, exist_ok=True)
                (S.RUN / "controls" / f"{ds}.json").write_text(json.dumps(ctl, indent=1))
        status["complete"] = True
    except StopIteration as e:
        status["complete"] = False
        log(f"[{ds}] budget stop before {e}")
    (S.RUN / f"STATUS_{ds}.json").write_text(json.dumps(status, indent=1, default=str))
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(S.CELLS))
    ap.add_argument("--lock", required=True)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from .lock import verify_lock
    lock = json.loads(Path(a.lock).read_text())
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: execution lock does not verify: " + "; ".join(v["mismatches"][:10]))
    run_dataset(a.dataset, lock)


if __name__ == "__main__":
    main()
