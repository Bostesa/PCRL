"""[lra port of lcr/run.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Runner for the learned-decoder constrained-release study (lra; atomic hash-verified units; resumable; <= 2 workers).

Adapted from cbp/run.py at 7f3ec67 (same unit format, same save/resume and lock checks).

    <PRIVATE_CACHE>/lra_v1/run/work.sh <LOCK> <stage> [i/n]
      == python -m lra.sema --label A:<stage> -- env OMP_NUM_THREADS=1 ... <python> -m lra.run --lock <LOCK> --stage ...

Stages (each refuses unless the latest named lock + amendments verify and are on origin, and every study module the
stage loads is locked with its hash -- lra.lock.check_loaded_modules):
  SOURCE_ADMISSION_LOCK  admit      verified copies of the cbp teachers, references, fine partitions, the D0 bank
                                    (DIRECT-TASK, FINE-TASK, CLASS-ONLY, 24 privacy maps x 3 seeds) and their cbp inner
                                    audits + teacher forward parity + release re-encode parity (lra.admit)
  CORRECTNESS_LOCK       correctness the engineering-readiness gate on the four pinned lcr fixture laws (lra.fixtures)
  SCIENCE_LOCK           d1         D1 learned decoders on the EXACT fixed D0 maps (calibration-only controls)
                         ctask      C-TASK per seed (3 units; every other new arm depends on it)
                         fit        72 weighted controls + 15 constrained units as 21 dependency chains (lra.mapper)
                         inner      inner audits of every new release (lra.audit)
                         inner_src  continuous-source inner audits composed over the COMPLETE registered bank
                         controls   real-data null and planted controls
                         select     lra.select.select_all
Assessment labels stay sealed (lra.data); only lra.assess unseals, after the pushed EVALUATION_LOCK.

CONFIG IDS (shared contract; every module uses these helpers):
  D0 (admitted, unchanged)   U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64, U|CLASS|i1o1, U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lam}
  D1 fixed-map controls      <D0 id>|D1 for DIRECT-TASK, FINE-TASK and the 24 privacy maps (assignments unchanged)
  C-TASK                     U|C-TASK|i8o64|D1
  weighted controls          U|W-{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lam}|D1
  constrained arms           U|K-{LOCAL,SEQ-12,SEQ-21,JOINT-SINGLE,JOINT-PAIR}|i8o64|D1
  sources / references       SRC|U, SRC|RAW-J_b0.3, REF|E, REF|F, REF|F0
UNITS: pol__s{k}__<safe> (D0, admitted), dec__s{k}__<safe> (D1 fixed-map), new__s{k}__<safe> (fitted), tea__, ref__,
fine__; lra inner audits aud__<unit> (run.inner_name); admitted cbp inner__<unit> audits are custody only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "lra_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_adult_learned_decoder_release_v1"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
RATE = (8, 64)
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
CONSTRAINED = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
LATE = {"d1": ("lra.run", "stage_d1"), "ctask": ("lra.run", "stage_ctask"), "fit": ("lra.run", "stage_fit"), "inner": ("lra.audit", "stage_inner"),
        "inner_src": ("lra.audit", "stage_inner_src"), "controls": ("lra.audit", "stage_controls"),
        "select": ("lra.select", "select_all"), "d0same": ("lra.select", "stage_d0same"),
        "correctness": ("lra.fixtures", "stage_correctness")}


def g(x):
    return f"{x:g}"


# ------------------------------------------------------------------ config ids
def d0_id(family, lam=None):
    if family == "CLASS":
        return "U|CLASS|i1o1"
    if family in ("DIRECT-TASK", "FINE-TASK"):
        return f"U|{family}|i8o64"
    return f"U|{family}|i8o64|l{g(lam)}"


def d1_id(family, lam=None):
    return d0_id(family, lam) + "|D1"


def ctask_id():
    return "U|C-TASK|i8o64|D1"


def weighted_id(family, lam):
    return f"U|W-{family}|i8o64|l{g(lam)}|D1"


def constrained_id(arm):
    assert arm in CONSTRAINED
    return f"U|K-{arm}|i8o64|D1"


def d0_ids():
    return [d0_id("DIRECT-TASK"), d0_id("FINE-TASK"), d0_id("CLASS")] + [d0_id(f, lam) for lam in LAMS for f in PRIVACY]


def d0_privacy_ids():
    return [d0_id(f, lam) for lam in LAMS for f in PRIVACY]


def d1_fixed_ids():
    # CLASS|D1 is a registered ADDITION to the prompt's 26 per seed (role F R-1; prompt section 10 "do not remove CLASS
    # from the comparator pool ... measure that [after D1]"): 27 per seed, 81 units, 84 codes per seed
    return [d1_id("DIRECT-TASK"), d1_id("FINE-TASK"), d1_id("CLASS")] + \
        [d1_id(f, lam) for lam in LAMS for f in PRIVACY]


def weighted_ids():
    return [weighted_id(f, lam) for lam in LAMS for f in PRIVACY]


def constrained_ids():
    return [constrained_id(a) for a in CONSTRAINED]


def new_fit_ids():
    """The 30 new mapping-pair fits per seed (90 total): C-TASK, 24 weighted controls, 5 constrained arms."""
    return [ctask_id()] + weighted_ids() + constrained_ids()


def code_ids():
    """Every code release of the registered bank (one seed): D0 (27), D1 fixed-map (26), new fits (30) = 83."""
    return d0_ids() + d1_fixed_ids() + new_fit_ids()


def scored_ids():
    """Every inner-selection candidate / comparator (one seed): the code bank, both continuous sources, 3 references."""
    return code_ids() + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


D0SAME = "|D0SAME"


def parse_id(cid):
    if cid.endswith(D0SAME):
        # same-map D0 diagnostic (prompt section 13 stage 5): the mean-teacher decoded version of the EXACT map of a
        # D1 release; label-free; never a candidate, never composed, never audited for nomination
        base = parse_id(cid[:-len(D0SAME)])
        return {**base, "arm": "d0_same", "decoder": "D0", "of": cid[:-len(D0SAME)], "diagnostic_only": True}
    if cid.startswith("SRC|"):
        return {"kind": "source", "teacher": cid.split("|")[1]}
    if cid.startswith("REF|"):
        return {"kind": "reference", "label": cid.split("|")[1]}
    parts = cid.split("|")
    t, fam, rate = parts[0], parts[1], parts[2]
    rest = parts[3:]
    decoder = "D1" if rest and rest[-1] == "D1" else "D0"
    lam = next((float(x[1:]) for x in rest if x.startswith("l")), None)
    m1, m2 = rate[1:].split("o")
    if fam.startswith("W-"):
        arm, base = "weighted", fam[2:]
    elif fam.startswith("K-"):
        arm, base = "constrained", fam[2:]
    elif fam == "C-TASK":
        arm, base = "ctask", fam
    elif decoder == "D1":
        arm, base = "d1_fixed", fam
    else:
        arm, base = "d0", fam
    return {"kind": "policy", "teacher": t, "family": fam, "base_family": base, "arm": arm, "decoder": decoder,
            "m1": int(m1), "m2": int(m2), "lam": lam,
            "privacy_trained": base in PRIVACY + CONSTRAINED and fam != "C-TASK"}


def safe(cid):
    return cid.replace("|", "_")


def unit_for(k, cid):
    p = parse_id(cid)
    if p["kind"] == "source":
        return f"tea__s{k}__{p['teacher']}"
    if p["kind"] == "reference":
        return f"ref__s{k}__{p['label']}"
    prefix = {"d0": "pol", "d1_fixed": "dec", "d0_same": "d0s"}.get(p["arm"], "new")
    return f"{prefix}__s{k}__{safe(cid)}"


def inner_name(k, cid):
    """lra inner-audit unit of a release (namespace aud__; the admitted cbp inner__* audits stay untouched as custody and
    are never read as lra records)."""
    return f"aud__{unit_for(k, cid)}"


# ------------------------------------------------------------------ units, events, compute ledger
def _finite(o):
    """Recursively replace nonfinite floats by None so every new JSON is finite-or-null (prompt sec. 5)."""
    import math
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    return o


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw},
                           allow_nan=False) + "\n")


def ledger(stage, shard, wall, cpu):
    with open(RUN / "COMPUTE_LEDGER.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "shard": shard,
                            "pid": os.getpid(), "wall_s": wall, "cpu_s": cpu, "maxrss_bytes":
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    from jcv.finalize import unit_complete
    return unit_complete(U(name))


def rec(name):
    return json.loads((U(name) / "record.json").read_text())


def npz(name, file):
    z = np.load(U(name) / file)
    return {x: z[x] for x in z.files}


def save(name, files, record):
    """files: {fname: callable(path) | dict of arrays (-> npz) | dict/list (-> json) | str (-> text)}."""
    from jcv.finalize import save_unit

    def writer(v, fname):
        if callable(v):
            return v
        if isinstance(v, dict) and fname.endswith(".npz"):
            return lambda p: np.savez_compressed(p, **{k: np.asarray(a) for k, a in v.items()})
        if isinstance(v, (dict, list)):
            return lambda p: p.write_text(json.dumps(_finite(v), allow_nan=False))
        return lambda p: p.write_text(str(v))
    save_unit(U(name), {f: writer(v, f) for f, v in files.items()}, _finite(record))
    event("unit complete", unit=name)


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def teacher(k, t="U"):
    return npz(f"tea__s{k}__{t}", "teacher.npz")


def fit_rows(D):
    return np.asarray(D["idx"]["DEFENSE_FIT"])


def bind_meta(k, cid, D, t="U"):
    from qpc import deploy as DP
    trec = rec(f"tea__s{k}__{t}")
    return {"teacher": t, "seed": k, "config": cid, "teacher_model_sha256": trec["model_sha256"],
            "feature_names_sha256": DP.schema_sha256([str(x) for x in D["feature_names"]])}


# ------------------------------------------------------------------ admission (SOURCE_ADMISSION_LOCK)
def stage_admit(D, shard_spec=None):
    from lra import admit as AD
    r = AD.run(D)
    event("admission", verdict=r["verdict"], copied=len(r["copied_now"]), teachers=len(r["teachers"]),
          codes=len(r["codes"]))


# ------------------------------------------------------------------ fitting data (SCIENCE_LOCK)
def fit_data(D):
    """OSF_DEFENSE_FIT rows, true task labels and SEX read through the allowlist (procedure "fitting"; the NEW supervised
    use of this study, disclosed in EXPOSURE_LEDGER.md)."""
    from lra import data as DA
    tr = np.asarray(DA.labels_for(D, "fitting", "OSF_DEFENSE_FIT"))
    assert np.array_equal(tr, fit_rows(D))
    Y = {1: np.asarray(D["y"]["income"])[tr].astype(np.int64), 2: np.asarray(D["y"]["occupation_group"])[tr].astype(np.int64)}
    S = np.asarray(D["sex"])[tr].astype(np.int64)
    assert (S >= 0).all() and all((y >= 0).all() for y in Y.values())
    return tr, Y, S


def plugin_mi(a, s, b=None):
    """Plug-in MI (nats) of binary s with the categorical a (or the aligned pair (a, b))."""
    a = np.asarray(a, dtype=np.int64)
    if b is not None:
        a = a * (int(np.max(b)) + 1) + np.asarray(b, dtype=np.int64)
    _, inv = np.unique(a, return_inverse=True)
    J = np.zeros((inv.max() + 1, 2))
    np.add.at(J, (inv, np.asarray(s, dtype=np.int64)), 1.0)
    P = J / J.sum()
    pa, ps = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    m = P > 0
    return float(np.sum(P[m] * np.log(P[m] / (pa @ ps)[m])))


def policy_dict(name):
    return json.loads((U(name) / "policy.json").read_text())


CLASS_D1 = "U|CLASS|i1o1|D1"


def d1_jobs():
    return [(k, c) for k in SEEDS for c in d1_fixed_ids()]


def stage_d1(D, shard_spec=None):
    """D1 learned decoders on the EXACT admitted D0 maps (calibration-only controls; assignments unchanged)."""
    from lra import decoder as DEC
    from qpc import release as RL
    tr, Y, S = fit_data(D)
    for k, cid in shard(d1_jobs(), shard_spec):
        n = unit_for(k, cid)
        if done(n):
            continue
        t0, c0 = time.time(), time.process_time()
        d0 = cid[:-3]
        d0u = unit_for(k, d0)
        pair = RL.load_policy(U(d0u) / "policy.json")
        T = teacher(k)
        decs, fitstats = [], {}
        for i, pol in ((1, pair.p1), (2, pair.p2)):
            P = T[f"p{i}"][tr]
            tok, _, _ = RL.encode(pol, P, T[f"d{i}"][tr])
            dec = DEC.decode_policy(pol, tok, P, Y[i], config=cid, meta={"seed": k, "d0_unit": d0u})
            decs.append(dec)
            L0, B0 = DEC.fit_losses(dec.y, np.asarray(pol.token_proto, dtype=np.float64), len(tr))
            L1, B1 = DEC.fit_losses(dec.y, dec.q, len(tr))
            Lu, Bu = DEC.row_losses(P, Y[i])
            fitstats[str(i)] = {"L_U": Lu, "B_U": Bu, "L_D0": L0, "B_D0": B0, "L_D1": L1, "B_D1": B1, "tokens": int(pol.T),
                                "occupied_fit": int(np.unique(tok).size), "I_fit": plugin_mi(tok, S)}
        rel = DEC.release_arrays_d1(pair, decs[0], decs[1], T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
        z0 = npz(d0u, "release.npz")
        same_tok = all(np.array_equal(rel[f"tok{i}"], z0[f"tok{i}"]) for i in (1, 2))
        if not same_tok:
            raise AssertionError(f"{cid}: D1 tokens differ from the admitted D0 release (assignments must be unchanged)")
        body = DEC.decoder_pair_dict(cid, pair, decs[0], decs[1])
        I12 = plugin_mi(rel["tok1"][tr], S, rel["tok2"][tr])
        r = {"schema": "lra-d1-fixed-v1", "config": cid, "seed": k, "d0_config": d0, "d0_unit": d0u,
             "cfg": parse_id(cid), "policy_pair_fingerprint": pair.fingerprint(), "decoder_sha256": body["decoder_sha256"],
             "tokens_bitwise_equal_d0": same_tok, "fitting": fitstats, "I12_fit": I12,
             "certificates": {str(i): DEC.certificate_summary(d) for i, d in ((1, decs[0]), (2, decs[1]))},
             "note": "assignments unchanged: full-token information identical to the D0 release by construction",
             "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        save(n, {"release.npz": rel, "decoder.json": body, "policy.json": (U(d0u) / "policy.json").read_text()}, r)


def _starts(k, cids):
    return {c: policy_dict(unit_for(k, c)) for c in cids}


def stage_ctask(D, shard_spec=None):
    from lra import mapper as MP
    tr, Y, S = fit_data(D)
    for k in shard(list(SEEDS), shard_spec):
        cid = ctask_id()
        n = unit_for(k, cid)
        if done(n):
            continue
        fine = json.loads((U(f"fine__s{k}") / "fine.json").read_text())
        t0, c0 = time.time(), time.process_time()
        r, files = MP.fit_unit("C-TASK", fine, teacher(k), tr, Y, S, starts=_starts(k, [d0_id("FINE-TASK")]),
                               meta=bind_meta(k, cid, D))
        r.update({"seed": k, "config": cid, "cfg": parse_id(cid), "wall_s": time.time() - t0,
                  "cpu_s": time.process_time() - c0, "origin": "NEW_FIT"})
        save(n, files, r)


def fit_chains():
    """21 dependency chains (run in order inside a chain): per (seed, lambda) the weighted W-LOCAL, W-SEQ-12, W-SEQ-21,
    W-JOINT; per seed the constrained K-LOCAL, K-SEQ-12, K-SEQ-21, K-JOINT-SINGLE, K-JOINT-PAIR. Constrained chains first
    (heaviest), then weighted, so the two shards balance."""
    chains = [[(k, constrained_id(a)) for a in CONSTRAINED] for k in SEEDS]
    chains += [[(k, weighted_id(f, lam)) for f in PRIVACY] for k in SEEDS for lam in LAMS]
    return chains


def _fit_args(k, cid):
    """(arm, lam, starts, witnesses, refs) for one new unit, exactly as registered in SEARCH_RULES.json."""
    from lra import mapper as MP
    p = parse_id(cid)
    src_priv = [d0_id(f, lam) for lam in LAMS for f in PRIVACY]
    src_fine = [d0_id("FINE-TASK")]
    ct = ctask_id()
    if p["arm"] == "constrained":
        a = p["base_family"]
        refs = MP.refs_from_ctask(rec(unit_for(k, ct)))
        if a == "LOCAL":
            return "K-LOCAL", None, _starts(k, [ct]), None, refs
        if a in ("SEQ-12", "SEQ-21"):
            return f"K-{a}", None, _starts(k, [ct] + [d0_id(a, lam) for lam in LAMS]), None, refs
        wit = [ct, constrained_id("LOCAL"), constrained_id("SEQ-12"), constrained_id("SEQ-21")] + src_fine + src_priv
        return f"K-{a}", None, None, _starts(k, wit), refs
    a, lam = p["base_family"], p["lam"]
    if a == "LOCAL":
        return "W-LOCAL", lam, _starts(k, [ct]), None, None
    if a in ("SEQ-12", "SEQ-21"):
        return f"W-{a}", lam, _starts(k, [ct] + [d0_id(a, l2) for l2 in LAMS]), None, None
    wit = [ct, weighted_id("LOCAL", lam), weighted_id("SEQ-12", lam), weighted_id("SEQ-21", lam)] + src_fine + src_priv
    return "W-JOINT", lam, None, _starts(k, wit), None


def stage_fit(D, shard_spec=None):
    from lra import mapper as MP
    tr, Y, S = fit_data(D)
    missing = [unit_for(k, ctask_id()) for k in SEEDS if not done(unit_for(k, ctask_id()))]
    if missing:
        raise SystemExit(f"REFUSED: run the ctask stage first; missing {missing}")
    fines = {k: json.loads((U(f"fine__s{k}") / "fine.json").read_text()) for k in SEEDS}
    for chain in shard(fit_chains(), shard_spec):
        for k, cid in chain:
            n = unit_for(k, cid)
            if done(n):
                continue
            arm, lam, starts, wit, refs = _fit_args(k, cid)
            t0, c0 = time.time(), time.process_time()
            r, files = MP.fit_unit(arm, fines[k], teacher(k), tr, Y, S, lam=lam, starts=starts, witnesses=wit,
                                   refs=refs, meta=bind_meta(k, cid, D))
            r.update({"seed": k, "config": cid, "cfg": parse_id(cid), "wall_s": time.time() - t0,
                      "cpu_s": time.process_time() - c0, "origin": "NEW_FIT"})
            save(n, files, r)


# ------------------------------------------------------------------ main
SCIENCE_STAGES = ("d1", "ctask", "fit", "inner", "inner_src", "controls", "select", "d0same")
GATE_RESULT = "ENGINEERING_GATE_RESULT.json"


def engineering_ready():
    """Prompt section 9 / finding 10: every Adult science stage requires the correctness stage's ENGINEERING_READY
    result, pushed to origin and bound (sha256) into the latest named lock's documents. Never hard-coded True."""
    from lra import lock as LK
    p = PKG / GATE_RESULT
    if not p.exists():
        return False, f"{GATE_RESULT} missing"
    r = json.loads(p.read_text())
    if r.get("verdict") != "ENGINEERING_READY":
        return False, f"verdict is {r.get('verdict')!r}, not ENGINEERING_READY"
    lat = LK.latest()
    if lat["documents_sha256"].get(GATE_RESULT) != LK.sha_file(p):
        return False, f"{GATE_RESULT} is not the version bound in {lat['name']}"
    if not LK.on_origin(f"{LK.REL}/{GATE_RESULT}"):
        return False, f"{GATE_RESULT} is not on origin"
    return True, "ENGINEERING_READY"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from lra import lock as LK
    v = LK.verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    if a.stage in SCIENCE_STAGES:
        ok, why = engineering_ready()
        if not ok:
            raise SystemExit("REFUSED: Adult science stages require ENGINEERING_READY: " + why)
    D = None
    if a.stage != "correctness":                 # fixtures are synthetic known laws: no real data is loaded
        from lra import data as DA
        D = DA.load()
        assert D["sealed"]
    fn = {"admit": stage_admit}
    if a.stage in fn:
        f = fn[a.stage]
    elif a.stage in LATE:
        mod, name = LATE[a.stage]
        f = getattr(importlib.import_module(mod), name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    for m in {"admit": ["lra.admit"], "correctness": ["lra.fixtures", "lra.decoder", "lra.mapper"],
              "d1": ["lra.decoder"], "ctask": ["lra.mapper", "lra.decoder"], "fit": ["lra.mapper", "lra.decoder"], "inner": ["lra.audit"],
              "inner_src": ["lra.audit"], "controls": ["lra.audit"], "select": ["lra.select"]}.get(a.stage, []):
        importlib.import_module(m)
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED: unlocked or changed code loaded: " + "; ".join(bad[:10]))
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid(), lock=v["lock"])
    t0, c0 = time.time(), time.process_time()
    f(D, a.shard)
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED after stage (lazy import of unlocked code): " + "; ".join(bad[:10]))
    ledger(a.stage, a.shard, time.time() - t0, time.process_time() - c0)
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
