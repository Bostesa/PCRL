"""Runner for the output-leak diagnosis study (stages 2-4). One worker per dataset.

    python -m odx.run --dataset adult --lock results/combined_output_diagnosis_v1/LOCK.json [--stage s2s3|s4|controls]

Units live in ~/PCRL_eval_cache_private/odx_v1/run/units/<uid>/ (preds.npz, val_preds.npz, models/, record.json,
COMPLETE.json). Exactly matching output-aware units are reused through ALIAS.json records (no copy, hash-checked).
Fixed order: datasets -> seeds 0,1,2 -> pairs in PAIRS order -> surfaces in SURFACE_ORDER -> bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

import oar.study as S
from .surfaces import surfaces, onehot

HOME = Path.home()
OAR_UNITS = HOME / "PCRL_eval_cache_private" / "oar_v1" / "run" / "units"
RUN = HOME / "PCRL_eval_cache_private" / "odx_v1" / "run"
S.RUN = RUN            # every unit written by the reused oar.study helpers goes to this study's run directory

PAIRS = [  # (dataset, purpose, attribute) - the 14 admitted pairs, fixed order
    ("adult", "income_prediction", "sex"), ("adult", "income_prediction", "race"),
    ("adult", "employment_analysis", "race"), ("adult", "employment_analysis", "age_group"),
    ("adult", "employment_analysis", "marital_status"),
    ("adult", "education_assessment", "sex"), ("adult", "education_assessment", "race"),
    ("adult", "education_assessment", "income"),
    ("hmda", "underwriting", "race"), ("hmda", "underwriting", "ethnicity"),
    ("hmda", "pricing_analysis", "race"), ("hmda", "pricing_analysis", "sex"),
    ("hmda", "fair_lending_audit", "race"), ("hmda", "fair_lending_audit", "sex")]
PRIMARY = {("adult", "income_prediction", "sex"), ("hmda", "underwriting", "race")}
SURFACE_ORDER = ("full", "dc", "centred", "prob", "offset", "hard")
BANK_CANDIDATES = ("full", "dc", "centred", "prob")          # full bank; validation log loss (NL, attacker seed 0); ties -> earlier listed
IO_BANK_CANDIDATES = ("centred", "prob")                     # ignore-offset bank (offset-free surfaces only)
OFFSET_USING = ("full", "dc")
REUSE_OAR = {"full": "O_full", "prob": "O_prob", "hard": "O_hard"}  # frozen-head surfaces of the primary cells
COALITION = {"dataset": "adult", "purposes": ("income_prediction", "employment_analysis"), "attribute": "race",
             "contracts": ("full", "centred", "hard")}


def purposes(ds):
    idx = json.loads((S.BENCH / "inputs" / "INPUTS_INDEX.json").read_text())
    return idx["datasets"][ds]["purposes"]


def K_of(ds, attr):
    L = np.load(S.BENCH / "inputs" / f"{ds}_labels.npz")
    return int(L[attr].max()) + 1


def world(ds, purpose, attr):
    """oar-roles-v1 roles (repaired loader) with this pair's sensitive attribute and the purpose's task."""
    W = S.load_world(ds)
    L = np.load(S.BENCH / "inputs" / f"{ds}_labels.npz")
    p = purposes(ds)[purpose]
    W = dict(W)
    W["s"], W["t"] = L[attr].astype(int), L[p["labels_task_key"]].astype(int)
    return W


def uid_of(ds, k, purpose, attr, stratum, surface):
    return f"{ds}__s{k}__{purpose}__{attr}__{stratum}__{surface}"


def _complete(d: Path) -> bool:
    c = d / "COMPLETE.json"
    if not c.exists():
        return False
    for f, h in json.loads(c.read_text())["files"].items():
        if not (d / f).exists() or hashlib.sha256((d / f).read_bytes()).hexdigest() != h:
            return False
    return True


def write_alias(uid, source_dir: Path, why: str):
    """Reuse an exactly matching historical unit: record its path and COMPLETE.json hash; copy record.json only."""
    d = S.unit_dir(uid)
    if _complete(d):
        return
    if not _complete(source_dir):
        raise RuntimeError(f"alias source {source_dir.name} is not hash-complete")
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "record.json", d / "record.json")
    src_c = hashlib.sha256((source_dir / "COMPLETE.json").read_bytes()).hexdigest()
    (d / "ALIAS.json").write_text(json.dumps({"source": "~/" + str(source_dir.relative_to(HOME)),
                                              "source_COMPLETE_sha256": src_c, "why": why}, indent=1))
    files = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("record.json", "ALIAS.json")}
    (d / "COMPLETE.json").write_text(json.dumps({"id": uid, "files": files, "alias": True}, indent=1))


def bank_unit(uid, candidates: list, note: str):
    """Validation-only selection among already complete candidate units (no new fit)."""
    d = S.unit_dir(uid)
    if _complete(d):
        return json.loads((d / "record.json").read_text())
    vals, resolved = {}, {}
    for c in candidates:
        r = json.loads((S.unit_dir(c) / "record.json").read_text())
        vals[c] = float(r["val_log_loss"]["NL"])
        resolved[c] = r.get("bank_selected") or c          # a nested bank resolves to its own selection
    best = None
    for c in candidates:
        if best is None or vals[c] < vals[best]:
            best = c
    sel = resolved[best]
    rec = {"id": uid, "kind": "bank", "candidates_attacker_val_log_loss": vals, "bank_selected": sel,
           "val_log_loss": {"NL": vals[best]}, "selection_role": "attacker_val", "note": note}
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps(rec, indent=1))
    (d / "COMPLETE.json").write_text(json.dumps({"id": uid, "files": {"record.json": hashlib.sha256(
        (d / "record.json").read_bytes()).hexdigest()}}, indent=1))
    return rec


def ledger(ds):
    def f(uid, cpu, fits):
        RUN.mkdir(parents=True, exist_ok=True)
        with open(RUN / f"LEDGER_{ds}.jsonl", "a") as fh:
            fh.write(json.dumps({"uid": uid, "cpu_s": cpu, "fits": fits,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return f


def spent_cpu_s():
    return sum(json.loads(l)["cpu_s"] for p in RUN.glob("LEDGER_*.jsonl") for l in p.read_text().splitlines())


def event(msg):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "EVENTS.jsonl", "a") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg}) + "\n")


def run_pairs(ds, lock, log=print):
    E, auth, syn = S.effective(), S.auth_real(), False
    led = ledger(ds)
    budget = float(lock["runtime"]["runner_cpu_s_budget"])
    done = []

    def atk(uid, X, W, s, K, finite=False):
        if spent_cpu_s() > budget:
            raise StopIteration(uid)
        r = S.attack_unit(uid, X, W, s, K, E, auth, syn, finite=finite, cpu_ledger=led)
        done.append(uid)
        return r

    for k in S.SEEDS:
        for (d_, purpose, attr) in PAIRS:
            if d_ != ds:
                continue
            p = purposes(ds)[purpose]
            W = world(ds, purpose, attr)
            s, t, K = W["s"], W["t"], K_of(ds, attr)
            F = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
            assert np.array_equal(F["row_id"], W["row_id"])
            Lg = F[p["logits_key"]].astype(np.float64)
            # references (label-only P(s|true task label), constant prior)
            ref = uid_of(ds, k, purpose, attr, "REF", "ref")
            if (ds, purpose, attr) in PRIMARY:
                write_alias(ref, OAR_UNITS / f"{ds}__s{k}__REF", "identical roles, task, attribute and reference fit")
            else:
                S.reference_unit(ref, W, s, t, K, int(Lg.shape[1]), E, auth, syn)
            strata = {"FH": Lg}
            if (ds, purpose, attr) in PRIMARY:
                strata["RH"] = np.load(OAR_UNITS / f"{ds}__s{k}__HEAD__A" / "preds.npz")["head_outputs_all"]
            for st, Lmat in strata.items():
                sf = surfaces(Lmat)
                for name in SURFACE_ORDER:
                    uid = uid_of(ds, k, purpose, attr, st, name)
                    if st == "FH" and (ds, purpose, attr) in PRIMARY and name in REUSE_OAR:
                        write_alias(uid, OAR_UNITS / f"{ds}__s{k}__{REUSE_OAR[name]}",
                                    "identical surface, roles, attacker slate and seeds (output-aware study)")
                        continue
                    atk(uid, sf[name], W, s, K, finite=(name == "hard"))
                bank_unit(uid_of(ds, k, purpose, attr, st, "iobank"),
                          [uid_of(ds, k, purpose, attr, st, c) for c in IO_BANK_CANDIDATES],
                          "ignore-offset bank: validation-selected attacker on offset-free surfaces")
                bank_unit(uid_of(ds, k, purpose, attr, st, "fullbank"),
                          [uid_of(ds, k, purpose, attr, st, c) for c in BANK_CANDIDATES],
                          "full-output bank: offset-using candidates plus every ignore-offset candidate")
            log(f"[{ds}] s{k} {purpose}/{attr} done")
    event(f"{ds}: pairs complete ({len(done)} new attack units)")
    return done


def run_coalition(lock, log=print):
    """S4: Adult income + employment recipients, attribute race; 9 slots per seed."""
    E, auth, syn = S.effective(), S.auth_real(), False
    led = ledger("adult_s4")
    ds, (pa, pb), attr = COALITION["dataset"], COALITION["purposes"], COALITION["attribute"]
    for k in S.SEEDS:
        Wa, Wb = world(ds, pa, attr), world(ds, pb, attr)
        for key in ("row_id", "unit", "s"):
            assert np.array_equal(Wa[key], Wb[key])
        for r in Wa["idx"]:
            assert np.array_equal(Wa["idx"][r], Wb["idx"][r])
        F = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
        sa = surfaces(F[purposes(ds)[pa]["logits_key"]].astype(np.float64))
        sb = surfaces(F[purposes(ds)[pb]["logits_key"]].astype(np.float64))
        K = K_of(ds, attr)
        for c in COALITION["contracts"]:
            uid = f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}"
            S.attack_unit(uid, np.hstack([sa[c], sb[c]]), Wa, Wa["s"], K, E, auth, syn, finite=(c == "hard"),
                          cpu_ledger=led)
            single = lambda p_: uid_of(ds, k, p_, attr, "FH", {"full": "fullbank", "centred": "iobank"}.get(c, c))  # noqa: E731
            bank_unit(f"{uid}__bank", [uid, single(pa), single(pb)],
                      "pair attacker vs each single recipient's attacker (ignore-the-other candidates)")
        log(f"[S4] s{k} done")
    event("S4 coalition complete")


def run_controls(ds, lock):
    """Real-data null and planted-leak controls, seed 0, primary cell, frozen-head full and centred surfaces."""
    E, auth, syn = S.effective(), S.auth_real(), False
    (d_, purpose, attr) = next(c for c in PRIMARY if c[0] == ds)
    W = world(ds, purpose, attr)
    F = np.load(S.BENCH / "inputs" / f"{ds}_s0_forward.npz")
    sf = surfaces(F[purposes(ds)[purpose]["logits_key"]].astype(np.float64))
    out = {}
    for name in ("full", "centred"):
        t0 = time.process_time()
        out[name] = S.null_and_planted(f"{ds}__CTL__{name}", sf[name], W, W["s"], K_of(ds, attr), E, auth, syn)
        ledger(ds)(f"{ds}__CTL__{name}", time.process_time() - t0, 0)
    (RUN / "controls").mkdir(parents=True, exist_ok=True)
    (RUN / "controls" / f"{ds}.json").write_text(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("adult", "hmda"))
    ap.add_argument("--stage", default="s2s3", choices=("s2s3", "s4", "controls"))
    ap.add_argument("--lock", required=True)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from .lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    lock = json.loads(Path(a.lock).read_text())
    event(f"start {a.stage} {a.dataset}")
    if a.stage == "s2s3":
        run_pairs(a.dataset, lock)
    elif a.stage == "s4":
        run_coalition(lock)
    else:
        run_controls(a.dataset, lock)
    event(f"end {a.stage} {a.dataset}")


if __name__ == "__main__":
    main()
