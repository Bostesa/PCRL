"""Stage A: separate convergence from capacity (DIRECT-TASK k-means codes per teacher-predicted class).

Runner interface (qpc/run.py calls these; every unit returns (record JSON-safe dict, files {name: writer | arrays})):

    a1_unit(T, tr, src_release, meta)            A1: dpc 20-round reproduction at (8, 8) with parity against the
                                                 admitted dpc DIRECT-TASK m8 release, then the same source start
                                                 refitted with <= 200 rounds and the qpc convergence rule.
    recipient_fit(P_fit, d_fit, K, m, recipient) A2 per-recipient fit: three starts, <= 200 rounds, per-class winner
                                                 (cached by the runner per (seed, recipient, m)).
    pair_unit(pol1_dict, pol2_dict, T, meta)     assemble U|DIRECT-TASK|i{m1}o{m2} and encode ALL rows.

T = teacher dict over ALL rows (row_id, p1 (n, 2), p2 (n, 6), d1, d2); tr = DEFENSE_FIT row indices. Only the
teacher probabilities of DEFENSE_FIT rows enter any fit; no task label or SEX is read here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dpc import partition as DPT
from qpc import kmeans as KM
from qpc import release as RL

A1_M = 8
A1_ROUNDS_SRC = 20
RATES1 = (4, 8)
RATES2 = (8, 16, 32, 64)
KS = {1: 2, 2: 6}
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def config_id(teacher, family, m1, m2, lam=None):
    if family == "CLASS":
        return f"{teacher}|CLASS|i1o1"
    return f"{teacher}|{family}|i{int(m1)}o{int(m2)}" + (f"|l{lam:g}" if family in PRIVACY else "")


def rate_pairs():
    return [(m1, m2) for m1 in RATES1 for m2 in RATES2]


def _json_writer(obj):
    s = json.dumps(obj, allow_nan=False)
    return lambda p: Path(p).write_text(s)


def fit_distortion(pol: RL.Policy):
    """D = mean fitting KL(p || decoded) from the stored statistics: (sum_f A_f - sum_t S_t . log q_t) / N."""
    N = int(pol.fine.n.sum())
    if N == 0:
        return 0.0
    used = pol.token_n > 0
    h = -np.sum(pol.token_S[used] * np.log(pol.token_proto[used]))
    return float((float(np.sum(pol.fine.A)) + h) / N)


def _class_summary(fit: KM.RecipientFit):
    out = []
    for pc in fit.receipt["per_class"]:
        if pc["fallback"]:
            out.append({"class": pc["class"], "rows": 0, "fallback": True})
            continue
        out.append({"class": pc["class"], "rows": pc["rows"], "fallback": False, "winner": pc["winner"],
                    "cells": pc["cells"],
                    "starts": [{k: r[k] for k in ("start", "objective", "mean_kl", "rounds_used", "converged",
                                                  "stop_reason", "initial_cells", "empty_cell_events",
                                                  "removed_empty_cells", "cell_counts", "work")}
                               for r in pc["starts"]]})
    return out


# ----------------------------------------------------------------------------------------------- A2 recipient fits
def recipient_fit(P_fit, d_fit, K, m, recipient, *, starts=KM.STARTS, rounds=KM.ROUNDS):
    """Three-start direct k-means of one recipient at cap m. Returns (qpc.Policy dict with the identity map on the
    per-class winning partition, receipts with every per-class start result and the winner)."""
    if int(recipient) not in KS or KS[int(recipient)] != int(K):
        raise ValueError("recipient 1 is income (K=2), recipient 2 is occupation (K=6)")
    fit = KM.fit_recipient(P_fit, d_fit, K, m, starts=starts, rounds=rounds, tag=f"DIRECT|r{recipient}|m{m}")
    pol = RL.make_policy(recipient, fit.partition, None, "DIRECT-TASK", {"m": int(m), "stage": "A2"})
    rec = dict(fit.receipt)
    rec.update({"recipient": int(recipient), "policy_fingerprint": pol.fingerprint(),
                "fit_distortion": fit_distortion(pol), "tokens_per_class": pol.tokens_per_class(),
                "alphabet": pol.T})
    KM.json_safe(rec)
    return pol.to_dict(), rec


# ----------------------------------------------------------------------------------------------- pair assembly
def _bind_meta(meta):
    meta = dict(meta or {})
    for k in RL.BINDING_KEYS:
        if not (isinstance(meta.get(k), str) and len(meta[k]) == 64):
            raise ValueError(f"meta must carry {k} (sha256 hex) to bind the policy for deployment")
    return meta


def assemble_pair(pol1, pol2, family, m1, m2, lam=None, meta=None):
    pol1 = pol1 if isinstance(pol1, RL.Policy) else RL.Policy.from_dict(pol1)
    pol2 = pol2 if isinstance(pol2, RL.Policy) else RL.Policy.from_dict(pol2)
    return RL.make_pair(pol1, pol2, family, m1, m2, lam, meta)


def encode_all(pair: RL.PolicyPair, T):
    """dpc-format release arrays over ALL rows; raises on any class-preservation failure."""
    out = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    for i in (1, 2):
        if not np.array_equal(out[f"hard{i}"], np.asarray(T[f"d{i}"])):
            raise AssertionError(f"CLASS PRESERVATION FAILED on recipient {i}")
    return out


def pair_record(pair: RL.PolicyPair, out, tr=None):
    r = {"config": pair.config, "family": pair.family, "fingerprint": pair.fingerprint(),
         "alpha1": int(out["alpha1"]), "alpha2": int(out["alpha2"]),
         "class_preservation_all_rows": True, "rows": int(out["row_id"].shape[0]),
         "fit_distortion": {"D1": fit_distortion(pair.p1), "D2": fit_distortion(pair.p2)}}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        ft = None if tr is None else out[f"tok{i}"][tr]
        rr = RL.policy_receipt(pol, ft)
        rr["cell_population_per_class"] = [[int(x) for x in pol.token_n[pol.token_class == c]] for c in range(pol.K)]
        if ft is not None and not np.array_equal(np.bincount(ft, minlength=pol.T), pol.token_n):
            raise AssertionError(f"recipient {i}: fitting-row token counts differ from the stored statistics")
        r[f"r{i}"] = rr
    r["states_emitted_fit"] = {i: int(sum(r[f"r{i}"]["effective_states_per_class"])) for i in (1, 2)}
    r["total_states"] = int(r["states_emitted_fit"][1] + r["states_emitted_fit"][2])
    return r


def pair_unit(pol1_dict, pol2_dict, T, meta, tr=None):
    """U|DIRECT-TASK|i{m1}o{m2} from two cached recipient fits. meta: teacher, seed, config, teacher_model_sha256,
    feature_names_sha256 (bound into the policy). Files: policy.json, release.npz (ALL rows)."""
    t0, c0 = time.perf_counter(), time.process_time()
    meta = _bind_meta(meta)
    p1, p2 = RL.Policy.from_dict(pol1_dict), RL.Policy.from_dict(pol2_dict)
    m1, m2 = int(p1.meta["m"]), int(p2.meta["m"])
    cid = config_id(meta.get("teacher", "U"), "DIRECT-TASK", m1, m2)
    if meta.get("config") not in (None, cid):
        raise ValueError(f"meta config {meta.get('config')} differs from {cid}")
    pair = assemble_pair(p1, p2, "DIRECT-TASK", m1, m2, None, {**meta, "config": cid})
    RL.check_bound(pair)
    out = encode_all(pair, T)
    rec = pair_record(pair, out, tr)
    rec.update({"stage": "A2", "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0})
    KM.json_safe(rec)
    return rec, {"policy.json": _json_writer(pair.to_dict()), "release.npz": out}


# ----------------------------------------------------------------------------------------------- A1
def a1_fits(P1, d1, P2, d2):
    """(src20 fits, r200 fits) per recipient on fitting rows. src20 is asserted bitwise equal to
    dpc.partition.fit_fine(P, d, K, max_cells=8, rounds=20)."""
    src20, r200, parity_dpc = {}, {}, {}
    for r, P, d in ((1, P1, d1), (2, P2, d2)):
        K = KS[r]
        f20 = KM.fit_recipient(P, d, K, A1_M, starts=("source",), rounds=A1_ROUNDS_SRC, rule="dpc",
                               tag=f"A1|src20|r{r}")
        ref = DPT.fit_fine(P, d, K, max_cells=A1_M, rounds=A1_ROUNDS_SRC)
        same = bool(f20.partition.fingerprint() == ref.fingerprint() and np.array_equal(f20.partition.mean, ref.mean))
        if not same:
            raise AssertionError(f"A1 reproduction differs from dpc.partition.fit_fine on recipient {r}")
        parity_dpc[r] = {"fingerprint": ref.fingerprint(), "bitwise_equal_to_dpc_fit_fine": same}
        src20[r] = f20
        r200[r] = KM.fit_recipient(P, d, K, A1_M, starts=("source",), rounds=KM.ROUNDS, rule="qpc",
                                   tag=f"A1|r200|r{r}")
    return src20, r200, parity_dpc


def a1_unit(T, tr, src_release, meta):
    """A1 historical convergence diagnostic for one U seed. src_release: arrays of the admitted dpc
    pol__s{k}__U_DIRECT-TASK_m8/release.npz (row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2)."""
    t0, c0 = time.perf_counter(), time.process_time()
    meta = _bind_meta(meta)
    tr = np.asarray(tr)
    P1, d1, P2, d2 = (np.asarray(T[x])[tr] for x in ("p1", "d1", "p2", "d2"))
    src20, r200, parity_dpc = a1_fits(P1, d1, P2, d2)
    teacher = meta.get("teacher", "U")
    out, pairs, recs = {}, {}, {}
    for v, fits in (("src20", src20), ("r200", r200)):
        p1 = RL.make_policy(1, fits[1].partition, None, "DIRECT-TASK", {"m": A1_M, "stage": f"A1-{v}"})
        p2 = RL.make_policy(2, fits[2].partition, None, "DIRECT-TASK", {"m": A1_M, "stage": f"A1-{v}"})
        cid = config_id(teacher, "DIRECT-TASK", A1_M, A1_M)
        pair = RL.make_pair(p1, p2, "DIRECT-TASK", A1_M, A1_M, None,
                            {**meta, "config": cid, "a1_variant": v,
                             "rounds_cap": A1_ROUNDS_SRC if v == "src20" else KM.ROUNDS,
                             "rule": "dpc" if v == "src20" else "qpc"})
        RL.check_bound(pair)
        out[v] = encode_all(pair, T)
        pairs[v] = pair
        recs[v] = pair_record(pair, out[v], tr)
        recs[v]["per_recipient"] = {r: {"objective_total": fits[r].receipt["objective_total"],
                                        "mean_kl_fit": fits[r].receipt["mean_kl_fit"],
                                        "all_converged": fits[r].receipt["all_converged"],
                                        "work": fits[r].receipt["work"],
                                        "partition_fingerprint": fits[r].partition.fingerprint(),
                                        "per_class": _class_summary(fits[r])} for r in (1, 2)}
    # parity with the admitted dpc DIRECT-TASK m8 release (all rows)
    src = {k: np.asarray(v) for k, v in src_release.items()}
    parity = {"row_id_equal": bool(np.array_equal(src["row_id"], np.asarray(T["row_id"])))}
    for i in (1, 2):
        tp = RL.token_parity(out["src20"][f"tok{i}"], out["src20"][f"q{i}"], src[f"tok{i}"], src[f"q{i}"])
        tp["hard_equal"] = bool(np.array_equal(out["src20"][f"hard{i}"], src[f"hard{i}"]))
        tp["alpha_equal"] = bool(int(out["src20"][f"alpha{i}"]) == int(src[f"alpha{i}"]))
        tp.pop("bijection")
        parity[f"r{i}"] = tp
    parity["ok"] = bool(parity["row_id_equal"] and all(parity[f"r{i}"]["ok"] and parity[f"r{i}"]["hard_equal"]
                                                       and parity[f"r{i}"]["alpha_equal"] for i in (1, 2)))
    parity["q_rule"] = ("bitwise" if all(parity[f"r{i}"]["q_bitwise_equal"] for i in (1, 2))
                        else "max_abs_diff<=1e-15" if parity["ok"] else "MISMATCH")
    parity["ids_rule"] = ("exact_ids" if all(parity[f"r{i}"]["tokens_bitwise_equal"] for i in (1, 2))
                          else "bijection" if parity["ok"] else "MISMATCH")
    # fixed-rate convergence contrast on fitting rows (teacher KL only; no labels)
    contrast = {}
    for r in (1, 2):
        a, b = recs["src20"][f"fit_distortion"][f"D{r}"], recs["r200"]["fit_distortion"][f"D{r}"]
        contrast[f"D{r}"] = {"src20": a, "r200": b, "r200_minus_src20": b - a}
    rec = {"stage": "A1", "config": config_id(teacher, "DIRECT-TASK", A1_M, A1_M), "meta": meta,
           "dpc_fit_fine_reproduction": parity_dpc, "parity_with_admitted_release": parity,
           "src20": recs["src20"], "r200": recs["r200"], "fit_distortion_contrast": contrast,
           "rounds": {"src20": A1_ROUNDS_SRC, "r200": KM.ROUNDS}, "rtol": KM.RTOL, "patience": KM.PATIENCE,
           "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    KM.json_safe(rec)
    if not parity["ok"]:
        rec["ENGINEERING_BLOCKER"] = "A1 reproduction does not match the admitted dpc DIRECT-TASK m8 release"
    files = {"release_src20.npz": out["src20"], "release_r200.npz": out["r200"],
             "policy_src20.json": _json_writer(pairs["src20"].to_dict()),
             "policy_r200.json": _json_writer(pairs["r200"].to_dict())}
    return rec, files
