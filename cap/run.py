"""Useful-head matched comparison runner: Adult income_prediction / sex, encoder seeds 0,1,2; feature arms A (untreated),
B (official target LEACE), F (validation-selected official FARE nominee), F0 (matched zero-fairness tree), all reused by
hash from the output-aware study. Common release head = that study's HEAD__<arm> (benchmark U2 LR grid on defense_fit
minus the oar-head-v1 holdout, selected on the holdout by log loss; runtime input = the arm's features only).

    python -m cap.run --lock results/combined_analysis_paper_v1/LOCK.json [--stage fits|controls]
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
from odx.surfaces import surfaces

HOME = Path.home()
OAR = HOME / "PCRL_eval_cache_private" / "oar_v1" / "run"
RUN = HOME / "PCRL_eval_cache_private" / "cap_v1" / "run"
S.RUN = RUN
DS, PURPOSE, ATTR, K_S, K_T = "adult", "income_prediction", "sex", 2, 2
ARMS = {"A": "A", "B": "B", "F": "F", "F0": "FZ"}          # study arm -> output-aware tag
OUT_SURFACES = ("full", "dc", "centred", "prob", "hard")
FEAT_SURFACES = ("full", "centred", "prob", "hard")


def uid(k, arm, view, surface):
    return f"cap__s{k}__{arm}__{view}__{surface}"


def _complete(d: Path) -> bool:
    c = d / "COMPLETE.json"
    if not c.exists():
        return False
    return all((d / f).exists() and hashlib.sha256((d / f).read_bytes()).hexdigest() == h
               for f, h in json.loads(c.read_text())["files"].items())


def alias(new_uid, src: Path, why):
    d = S.unit_dir(new_uid)
    if _complete(d):
        return
    if not _complete(src):
        raise RuntimeError(f"alias source {src.name} not hash-complete")
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "record.json", d / "record.json")
    (d / "ALIAS.json").write_text(json.dumps({"source": "~/" + str(src.relative_to(HOME)), "why": why,
                                              "source_COMPLETE_sha256": hashlib.sha256((src / "COMPLETE.json").read_bytes()).hexdigest()}, indent=1))
    files = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("record.json", "ALIAS.json")}
    (d / "COMPLETE.json").write_text(json.dumps({"id": new_uid, "files": files, "alias": True}, indent=1))


def bank(new_uid, cands, note):
    d = S.unit_dir(new_uid)
    if _complete(d):
        return json.loads((d / "record.json").read_text())
    vals, res = {}, {}
    for c in cands:
        r = json.loads((S.unit_dir(c) / "record.json").read_text())
        vals[c] = float(r["val_log_loss"]["NL"])
        res[c] = r.get("bank_selected") or c
    best = min(cands, key=lambda c: (vals[c], cands.index(c)))
    rec = {"id": new_uid, "kind": "bank", "candidates_attacker_val_log_loss": vals, "bank_selected": res[best],
           "val_log_loss": {"NL": vals[best]}, "selection_role": "attacker_val", "note": note}
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps(rec, indent=1))
    (d / "COMPLETE.json").write_text(json.dumps({"id": new_uid, "files": {"record.json": hashlib.sha256((d / "record.json").read_bytes()).hexdigest()}}, indent=1))
    return rec


def features(k, arm, W, H):
    if arm == "A":
        return H
    if arm == "B":
        return S.leace_release(DS, k, "B", H)[0]
    sel = json.loads((OAR / "selection" / f"{DS}__s{k}.json").read_text())
    src = sel["nominee_unit_source"] if arm == "F" else None
    fit = OAR / "units" / (f"{DS}__s{k}__FAREFIT_c{src}" if arm == "F" else f"{DS}__s{k}__FAREFIT_Z")
    cells = np.load(fit / "cells.npy")
    return S.onehot(cells, int(cells.max()) + 1)


def features_only_source(k, arm):
    if arm == "F":
        sel = json.loads((OAR / "selection" / f"{DS}__s{k}.json").read_text())
        return OAR / "units" / f"{DS}__s{k}__Fc{sel['nominee_unit_source']}__rep"
    return OAR / "units" / f"{DS}__s{k}__{ARMS[arm]}__rep"


def ledger(uid_, cpu, fits):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "LEDGER.jsonl", "a") as f:
        f.write(json.dumps({"uid": uid_, "cpu_s": cpu, "fits": fits, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


def event(msg):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "EVENTS.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg}) + "\n")


def run_fits(log=print):
    E, auth, syn = S.effective(), S.auth_real(), False
    W = S.load_world(DS)
    s = W["s"]
    for k in S.SEEDS:
        H = S.load_seed(DS, k)["H"]
        for arm, tag in ARMS.items():
            X = features(k, arm, W, H)
            head = np.load(OAR / "units" / f"{DS}__s{k}__HEAD__{tag}" / "preds.npz")["head_outputs_all"]  # log-probabilities
            sf = surfaces(head)
            fin = arm in ("F", "F0")   # finite releases (tree cells): the cell-conditional attacker is also fitted (NLDA, descriptive)
            # features only (reuse), output-only full (reuse), features + own full output (reuse)
            alias(uid(k, arm, "feat", "none"), features_only_source(k, arm), "identical features-only unit (output-aware)")
            alias(uid(k, arm, "out", "full"), OAR / "units" / f"{DS}__s{k}__O_head{tag}", "identical own-head output-only unit")
            alias(uid(k, arm, "feat+out", "full"), OAR / "units" / f"{DS}__s{k}__{tag}__rep+head", "identical features + own head unit")
            alias(uid(k, arm, "feat+clean", "full"), OAR / "units" / f"{DS}__s{k}__{tag}__rep+clean", "historical clean-output bypass (adverse control)")
            for sname in OUT_SURFACES[1:]:
                S.attack_unit(uid(k, arm, "out", sname), sf[sname], W, s, K_S, E, auth, syn, finite=(sname == "hard") or fin,
                              contract={"view": "output-only", "surface": sname, "head": f"HEAD__{tag}"}, cpu_ledger=ledger)
            for sname in FEAT_SURFACES[1:]:
                S.attack_unit(uid(k, arm, "feat+out", sname), np.hstack([X, sf[sname]]), W, s, K_S, E, auth, syn,
                              finite=fin and sname == "hard",
                              plus={"rep_unit": uid(k, arm, "feat", "none"), "out_unit": uid(k, arm, "out", sname)},
                              contract={"view": "features+own-output", "surface": sname, "features": arm, "head": f"HEAD__{tag}"},
                              cpu_ledger=ledger)
            bank(uid(k, arm, "out", "iobank"), [uid(k, arm, "out", c) for c in ("centred", "prob")], "ignore-offset bank (output-only)")
            bank(uid(k, arm, "out", "fullbank"), [uid(k, arm, "out", c) for c in ("full", "dc", "centred", "prob")], "full-output bank (output-only)")
            bank(uid(k, arm, "feat+out", "iobank"), [uid(k, arm, "feat+out", c) for c in ("centred", "prob")], "ignore-offset bank (features + own output)")
            bank(uid(k, arm, "feat+out", "fullbank"), [uid(k, arm, "feat+out", c) for c in ("full", "centred", "prob")], "full bank (features + own output)")
            log(f"s{k} {arm} done")
    event("fits complete")


def run_controls():
    """Real-data null + planted-leak controls (attacker_fit/val only, seed 0) for each new interface type."""
    E, auth, syn = S.effective(), S.auth_real(), False
    W = S.load_world(DS)
    H = S.load_seed(DS, 0)["H"]
    out = {}
    for arm, tag, view, sname in (("A", "A", "out", "centred"), ("F", "F", "out", "hard"), ("F", "F", "feat+out", "hard"),
                                  ("B", "B", "out", "prob")):
        head = np.load(OAR / "units" / f"{DS}__s0__HEAD__{tag}" / "preds.npz")["head_outputs_all"]
        sf = surfaces(head)
        X = sf[sname] if view == "out" else np.hstack([features(0, arm, W, H), sf[sname]])
        t0 = time.process_time()
        out[f"{arm}/{view}/{sname}"] = S.null_and_planted(f"cap__CTL__{arm}__{view}__{sname}", X, W, W["s"], K_S, E, auth, syn,
                                                          finite=(sname == "hard"))
        ledger(f"cap__CTL__{arm}__{view}__{sname}", time.process_time() - t0, 0)
    (RUN / "controls.json").write_text(json.dumps(out, indent=1))
    event("controls complete")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", default="fits", choices=("fits", "controls"))
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from .lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    event(f"start {a.stage}")
    run_fits() if a.stage == "fits" else run_controls()


if __name__ == "__main__":
    main()
