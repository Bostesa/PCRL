"""Runner for the online-strength frontier study (atomic hash-verified units; resumable; 2-worker sharding).

    env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.run --lock <lock json> --stage <stage> [--shard i/n]

Stages and the named lock that governs them (osf.lock.STAGE_MIN_LOCK; each lock is pushed before its stages run):
  DATA_AND_ENGINEERING_LOCK  admit     admitted smf warm starts / U / RAW beta 0.1, 0.3 -> release units on the osf rows
                             parity    rho = 0 / beta = 0 == TASK (4 epochs, critics running), per seed
                             fidelity  RAW vs pinned rgj.train (2 epochs, real fitting rows) + frozen-minibatch
                                       equivalence on real rows
                             replay    instrumented 40-epoch replays of the 12 admitted RAW runs (+ 3 U): bitwise
                                       equality with the admitted checkpoints is required; receipts = their strength
                             timing    one logged 2-epoch NORM calibration fit (not a bank unit)
  TRAINING_PROTOCOL_LOCK     bank      every not-yet-admitted configuration of the locked bank, 3 seeds, 40 epochs
                             references  official LEACE / FARE / compression references (osf.baselines)
  SELECTION_AND_AUDIT_LOCK   inner     inner audits (osf.inner), select (osf.select), tracking (osf.track)
The consolidated assessment is never touched here (labels sealed at load; osf.assess unseals after the pushed
EVALUATION_LOCK).
Units (<PRIVATE_CACHE>/osf_v1/run/units): warm__s{k} (admitted copy); rel__s{k}__{cfg} (release: model.pt, heads,
release.npz on every osf row, critics.pt); run__s{k}__{cfg} (diag, steps.npz, final.pt, ck20.pt, captures.pt);
parity__*, fid__*, equiv__s{k}, timing__*.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv import train as JT
from osf import data as DA
from osf import train as T
from rgj import finalize as FN
from rgj import train as RT

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "osf_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_online_strength_frontier_v1"
SEEDS = (0, 1, 2)
ADMITTED_BETAS = (0.1, 0.3)
CAPTURE_CONFIGS = ("RAW-J|b0.3", "RAW-L|b0.3", "NORM-J|r3|a1", "NORM-L|r3|a1")
LATE = {"inner": ("osf.inner", "run_inner"), "select": ("osf.select", "select_all"),
        "tracking": ("osf.track", "run_tracking"), "references": ("osf.baselines", "run_references")}


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    return FN.unit_complete(U(name))


def rec(name):
    return json.loads((U(name) / "record.json").read_text())


def safe(cid):
    return cid.replace("|", "_")


def rel_name(k, cid):
    return f"rel__s{k}__{safe(cid)}"


def run_name(k, cid):
    return f"run__s{k}__{safe(cid)}"


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def model_from(state, k):
    m = T.Model(T.D_IN, T.KS, k)
    m.load_state_dict(state)
    return m


def smf_name(k, cid):
    """Admitted smf unit holding the epoch-40 checkpoint of an admitted configuration."""
    if cid == "U":
        return f"tl__s{k}__e40"
    c = T.parse_id(cid)
    return f"raw__s{k}__RAW-{c['treat']}__b{T.g(c['beta'])}__e40"


def admitted_ids():
    return ["U"] + [f"RAW-{t}|b{T.g(b)}" for t in ("J", "L") for b in ADMITTED_BETAS]


def load_warm(k):
    return torch.load(U(f"warm__s{k}") / "warm.pt")


def head(k):
    return T.head_of(load_warm(k))


def save_release_unit(name, state, k, D, record, critics=None, check_against=None):
    out, heads, meta = FN.finalize_model(model_from(state, k), D)
    if check_against is not None:                  # admitted model: must equal the smf release on every smf row
        record = {**record, "admitted_release_equal_on_smf_rows": admitted_equal(out, D, check_against)}
    files = {"model.pt": lambda p: torch.save(state, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    if critics is not None:
        files["critics.pt"] = lambda p: torch.save(critics, p)
    FN.save_unit(U(name), files, {**record, "unit": name, "seed": k, "heads": meta, "map": "identity (no erasure)",
                                  "model_sha256": T.state_sha(state)})
    event("unit complete", unit=name)


def save_run_unit(name, diag, steps, final, record, ck20=None, captures=None):
    files = {"final.pt": lambda p: torch.save(final, p),
             "steps.npz": lambda p: np.savez_compressed(p, **steps)}
    if ck20 is not None:
        files["ck20.pt"] = lambda p: torch.save(ck20, p)
    if captures:
        files["captures.pt"] = lambda p: torch.save(captures, p)
    FN.save_unit(U(name), files, {**record, "unit": name, "diag": diag})
    event("unit complete", unit=name)


def critics_of(snap):
    return {kk: snap[kk] for kk in ("critics", "transforms", "critic_head", "opts") if kk in snap}


def timed(fn, *a, **kw):
    t0, c0 = time.time(), time.process_time()
    out = fn(*a, **kw)
    return out, time.time() - t0, time.process_time() - c0


# ------------------------------------------------------------------ admission (DATA_AND_ENGINEERING_LOCK)
def stage_admit(D, shard_spec=None):
    from osf import admit as AD
    for k in shard(list(SEEDS), shard_spec):
        w = f"warm__s{k}"
        if not done(w):
            src = AD.admitted_path(w) / "warm.pt"
            st = torch.load(src)
            (rm, wall, cpu) = timed(JT.warm_start, T.D_IN, T.KS, RT.TData(D), k)     # bitwise warm-start replay
            same = eq_state(rm.state_dict(), st)
            if not same:
                raise SystemExit(f"ADMISSION FAILED {w}: the warm-start replay on OSF_DEFENSE_FIT differs")
            FN.save_unit(U(w), {"warm.pt": lambda p, st=st: torch.save(st, p)},
                         {"seed": k, "admitted_from": f"smf {w}", "warm_sha256": T.state_sha(st),
                          "replay_bitwise": same, "replay_wall_s": wall,
                          "schedule": "jcv.train.warm_start on OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT), admitted"})
            event("unit complete", unit=w)
        for cid in admitted_ids():
            n = rel_name(k, cid)
            if done(n):
                continue
            src = AD.admitted_path(smf_name(k, cid))
            st = torch.load(src / "model.pt")
            crit = torch.load(src / "critics.pt") if (src / "critics.pt").exists() else None
            save_release_unit(n, st, k, D, {"config": cid, "cfg": T.parse_id(cid), "epoch": 40, "admitted": True,
                                            "admitted_from": f"smf {smf_name(k, cid)}",
                                            "recipe": "smf task line (salt 0)" if cid == "U" else
                                            "rgj.train J-O/L-O unchanged, stage B (salt 0), 40 epochs"},
                              critics=crit, check_against=src)


def admitted_equal(out, D, src):
    """The rebuilt release must equal the admitted smf release bitwise on every row the smf release covers
    (checked before the unit is written; a mismatch refuses the admission)."""
    b = np.load(src / "release.npz")
    pos = {int(r): j for j, r in enumerate(D["row_id"])}
    ix = np.array([pos[int(r)] for r in b["row_id"]])
    ok = {key: bool(np.array_equal(np.asarray(out[key])[ix], b[key])) for key in b.files if key != "row_id"}
    if not all(ok.values()):
        raise SystemExit(f"ADMISSION FAILED {src.name}: rebuilt release differs from the admitted one: {ok}")
    event("admission check", source=src.name, ok=True, smf_rows=int(len(ix)))
    return {"smf_rows": int(len(ix)), "equal": ok}


# ------------------------------------------------------------------ engineering parity / fidelity / equivalence
PARITY = [{"mode": "RAW", "treat": "J", "beta": 0.0}, {"mode": "RAW", "treat": "L", "beta": 0.0},
          {"mode": "NORM", "treat": "J", "rho": 0.0, "a": 1.0}, {"mode": "NORM", "treat": "L", "rho": 0.0, "a": 2.0}]


def eq_state(a, b):
    return all(torch.equal(a[q], b[q]) for q in a) and set(a) == set(b)


def stage_parity(D, shard_spec=None):
    data = RT.TData(D)
    for k in shard(list(SEEDS), shard_spec):
        names = [f"parity__s{k}__{safe(T.config_id(c))}" for c in PARITY]
        if all(done(n) for n in names):
            continue
        warm = load_warm(k)
        tm, td, *_ = T.train_run({"mode": "TASK", "treat": None}, warm, data, k, None, n_epochs=4, ckpt_epochs=())
        for c, n in zip(PARITY, names):
            if done(n):
                continue
            m, d, *_ = T.train_run(c, warm, data, k, head(k), n_epochs=4, ckpt_epochs=())
            ok = eq_state(m.state_dict(), tm.state_dict())
            FN.save_unit(U(n), {}, {"seed": k, "config": T.config_id(c), "pass": bool(ok), "critic_online_updates":
                                    d["critic_online_updates"], "check": "strength 0 equals the task-only continuation "
                                    "from the same initialisation (4 epochs, salt 0), critics training on their own stream"})
            event("parity", unit=n, ok=bool(ok))
            if not ok:
                raise SystemExit(f"PARITY FAILED {n}")


def stage_fidelity(D, shard_spec=None):
    data = RT.TData(D)
    for k in shard(list(SEEDS), shard_spec):
        for t, arm in (("J", "J-O"), ("L", "L-O")):
            n = f"fid__s{k}__RAW-{t}_b0.3"
            if done(n):
                continue
            warm = load_warm(k)
            c = {"mode": "RAW", "treat": t, "beta": 0.3}
            m, d, ck, _, _, steps = T.train_run(c, warm, data, k, head(k), n_epochs=2, ckpt_epochs=(2,))
            m2, d2, ck2, _, _ = RT.train_run(arm, 0.3, warm, data, k, "B", n_epochs=2, critic_head=head(k),
                                             ckpt_epochs=(2,))
            same_model = eq_state(m.state_dict(), m2.state_dict())
            same_critics = all(eq_state(ck[2]["critics"][v][kk], ck2[2]["critics"][v][kk]) for v in RT.VIEWS
                               for kk in RT.KINDS)
            logged = [(e["step"], e["penalty"], e["task"]) for e in d2["norms"]]
            dev = max((max(abs(np.sqrt((steps["q_norm"][s - 1] ** 2).sum()) - p) / max(p, 1e-30),
                          abs(np.sqrt((steps["t_norm"][s - 1] ** 2).sum()) - tt) / max(tt, 1e-30)) for s, p, tt in logged), default=0.0)
            dev = float(dev) if logged else None    # rgj logs every 20 steps (6 entries on the real 2-epoch run)
            # frozen-minibatch equivalence on real rows, epoch-2 model (AMENDMENT_A1/A2). With the ONLINE critics of the
            # snapshot the constant wins every view on most minibatches (p_i = 0: the algebra check is vacuous); their
            # applicability over the whole epoch-2 order is recorded as a diagnostic. The check itself uses fresh,
            # deterministic bounded-refit critics on the frozen epoch-2 model (CRITIC_FIT fit, CRITIC_VAL early stop,
            # snapshot transform) and the first epoch-2 minibatch (salt 0 order) on which both proxy gradients are nonzero.
            perm = np.random.default_rng([k, T.HP["salt"], 2]).permutation(data.n)
            batches = [perm[s0:s0 + T.HP["batch"]] for s0 in range(0, data.n, T.HP["batch"])]
            online_scan = [T.frozen_equivalence(ck[2]["model"], data, k, head(k), t, 0.3, bi, snapshot=ck[2])["applicable"]
                           for bi in batches]
            refit = refit_snapshot(ck[2], data, k)
            skipped = []
            for j, bi in enumerate(batches):
                eqv = T.frozen_equivalence(ck[2]["model"], data, k, head(k), t, 0.3, bi, snapshot=refit)
                if eqv["applicable"] == 2:
                    break
                skipped.append({"batch_index": j, "applicable": eqv["applicable"]})
            eqv.update(batch_index=j, skipped_inapplicable_batches=skipped, critics="bounded refit on the frozen epoch-2 model",
                       refit_receipts=refit["receipts"],
                       online_critic_applicability={str(a): online_scan.count(a) for a in (0, 1, 2)})
            rr = [e["r_i"] for e in eqv["encoders"]]
            common = float(np.sqrt(np.nanmean(np.square(rr))))
            fail_common = T.frozen_equivalence(ck[2]["model"], data, k, head(k), t, 0.3, bi, snapshot=refit,
                                               rho_override=common)
            fail_cap = T.frozen_equivalence(ck[2]["model"], data, k, head(k), t, 0.3, bi, snapshot=refit,
                                            a_max=1e-6)
            close = abs(rr[0] - rr[1]) <= 10 * T.EQUIV_RTOL * max(rr)      # review A2: expected failures required
            ok = (same_model and same_critics and (dev is None or dev < 1e-5) and eqv["equivalent"]
                  and not fail_cap["equivalent"] and (not fail_common["equivalent"] or close))
            FN.save_unit(U(n), {}, {"seed": k, "config": T.config_id(c), "epochs": 2, "bitwise_model": same_model,
                                    "bitwise_critics": same_critics, "logged_norm_max_rel_dev": dev,
                                    "logged_norm_entries": len(logged),
                                    "equivalence": eqv, "expected_failure_common_rho": fail_common,
                                    "expected_failure_cap": fail_cap, "pass": bool(ok),
                                    "check": "osf RAW vs pinned rgj.train (identical warm state, fixed head, salt 0, "
                                    "critic init and streams) on OSF_DEFENSE_FIT; frozen-minibatch equivalence"})
            event("fidelity", unit=n, ok=bool(ok))
            if not ok:
                raise SystemExit(f"FIDELITY FAILED {n}")


def refit_snapshot(snap, data, k):
    """Fresh deterministic critics fitted on a FROZEN model (AMENDMENT_A2; fitting rows only): per view and kind,
    rgj.train.fit_bounded (Adam 3e-3, batch 256, <= 15 epochs, patience 3) on CRITIC_FIT with CRITIC_VAL early stopping,
    inputs = the snapshot's transform of the fixed-head views. For the algebra check only; never used in training."""
    model = model_from(snap["model"], k)
    V = RT.frozen_views(model, data.X, snap["critic_head"])
    cf, cv = torch.from_numpy(data.cf), torch.from_numpy(data.cv)
    crit, rec_ = {}, {}
    for v in RT.VIEWS:
        Tv = RT.Transform(state=snap["transforms"][v])
        Zf, Zv = Tv(V[v][cf]), Tv(V[v][cv])
        crit[v], rec_[v] = {}, {}
        for kind in RT.KINDS:
            c = RT.new_critic(k, v, kind, "equiv-refit")
            c, rc = RT.fit_bounded(c, Zf, data.S[cf], Zv, data.S[cv], [k, RT._seed(v, kind), 991])
            crit[v][kind], rec_[v][kind] = c.state_dict(), rc
    return {"critics": crit, "transforms": snap["transforms"], "receipts": rec_}


def stage_replay(D, shard_spec=None):
    """Instrumented replays of the admitted 40-epoch runs. The admitted checkpoint stays the released model; the replay
    must reproduce it (and its epoch-20 checkpoint) bitwise, and supplies the per-step strength receipts."""
    from osf import admit as AD
    data = RT.TData(D)
    jobs = [(k, cid) for k in SEEDS for cid in admitted_ids()]
    for k, cid in shard(jobs, shard_spec):
        rn = run_name(k, cid)
        if done(rn):
            continue
        c = T.parse_id(cid)
        caps = set(T.capture_steps_for(40, data.n).values()) if cid in CAPTURE_CONFIGS else ()
        (m, d, ck, cap, fin, steps), wall, cpu = timed(T.train_run, c, load_warm(k), data, k,
                                                       head(k) if c["mode"] != "TASK" else None, capture_steps=caps)
        src40 = AD.admitted_path(smf_name(k, cid))
        src20 = AD.admitted_path(smf_name(k, cid).replace("__e40", "__e20")) if cid != "U" else None
        match = {"e40": eq_state(ck[40]["model"], torch.load(src40 / "model.pt"))}
        if src20 is not None:
            match["e20"] = eq_state(ck[20]["model"], torch.load(src20 / "model.pt"))
        if c["mode"] == "RAW":
            a40 = torch.load(src40 / "critics.pt")
            match["e40_critics"] = all(eq_state(ck[40]["critics"][v][kk], a40["critics"][v][kk]) for v in RT.VIEWS
                                       for kk in RT.KINDS)
        save_run_unit(rn, d, steps, fin, {"config": cid, "cfg": c, "seed": k, "replay_of": f"smf {smf_name(k, cid)}",
                                          "bitwise": match, "wall_s": wall, "cpu_s": cpu}, ck20=ck.get(20),
                      captures=cap)
        event("replay", unit=rn, bitwise=match)
        if not all(match.values()):
            raise SystemExit(f"REPLAY MISMATCH {rn}: {match}")


def stage_timing(D, shard_spec=None):
    data = RT.TData(D)
    n = "timing__s0__NORM-J_r3_a1"
    if done(n):
        return
    c = {"mode": "NORM", "treat": "J", "rho": 3.0, "a": 1.0}
    (m, d, *_), wall, cpu = timed(T.train_run, c, load_warm(0), data, 0, head(0), n_epochs=2, ckpt_epochs=())
    FN.save_unit(U(n), {}, {"config": T.config_id(c), "seed": 0, "epochs": 2, "wall_s": wall, "cpu_s": cpu,
                            "s_per_epoch": wall / 2, "summary": d["summary"],
                            "role": "timing calibration only (logged; not a bank unit, never audited or selected)"})
    event("timing", unit=n, wall_s=wall)


# ------------------------------------------------------------------ the locked bank (TRAINING_PROTOCOL_LOCK)
def locked_bank():
    L = json.loads((PKG / "TRAINING_PROTOCOL_LOCK.json").read_text())
    kind = L["protocol"]["bank"]
    return kind, [T.config_id(c) for c in T.bank(kind)]


def bank_jobs():
    kind, ids = locked_bank()
    return [(k, cid) for k in SEEDS for cid in ids if cid not in admitted_ids()]


def stage_bank(D, shard_spec=None):
    data = RT.TData(D)
    for k, cid in shard(bank_jobs(), shard_spec):
        rn, n = run_name(k, cid), rel_name(k, cid)
        if done(rn) and done(n):
            continue
        c = T.parse_id(cid)
        caps = set(T.capture_steps_for(40, data.n).values()) if cid in CAPTURE_CONFIGS else ()
        (m, d, ck, cap, fin, steps), wall, cpu = timed(T.train_run, c, load_warm(k), data, k, head(k),
                                                       capture_steps=caps)
        if d["nonfinite"]:               # registered single half-lr retry, originals kept and labelled
            orig = run_name(k, cid) + "__nonfinite_original"
            if not done(orig):
                save_run_unit(orig, d, steps, fin, {"config": cid, "cfg": c, "seed": k, "status": "NONFINITE_ORIGINAL"})
            (m, d, ck, cap, fin, steps), wall, cpu = timed(T.train_run, c, load_warm(k), data, k, head(k),
                                                           capture_steps=caps, lr=T.HP["sgd_lr"] / 2)
            d["rescue"] = "half-lr retry after nonfinite (RESCUED; not the intended learning rate)"
        base = {"config": cid, "cfg": c, "seed": k, "wall_s": wall, "cpu_s": cpu, "rescued": "rescue" in d}
        if not done(n):
            save_release_unit(n, ck[40]["model"], k, D, {**base, "epoch": 40}, critics=critics_of(ck[40]))
        save_run_unit(rn, d, steps, fin, base, ck20=ck.get(20), captures=cap)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from osf.lock import verify_lock
    v = verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = DA.load()
    assert D["sealed"]
    if a.stage not in ("admit", "parity") and not parity_passed():
        raise SystemExit("REFUSED: engineering parity has not passed")
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid(), lock=v["lock"])
    fn = {"admit": stage_admit, "parity": stage_parity, "fidelity": stage_fidelity, "replay": stage_replay,
          "timing": stage_timing, "bank": stage_bank}
    if a.stage in fn:
        fn[a.stage](D, a.shard)
    elif a.stage in LATE:
        mod, f = LATE[a.stage]
        fn_late = getattr(importlib.import_module(mod), f)
        if a.stage == "references":                    # osf.baselines.run_references(D, seeds=..., shard=...)
            fn_late(D, shard=a.shard)
        else:
            fn_late(D, a.shard)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


def parity_passed():
    names = [f"parity__s{k}__{safe(T.config_id(c))}" for k in SEEDS for c in PARITY]
    return all(done(n) and rec(n)["pass"] for n in names)


if __name__ == "__main__":
    main()
