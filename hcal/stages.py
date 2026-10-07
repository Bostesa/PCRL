"""Science stages of the held-out calibration study (hcal; role A; frozen in SCIENCE_LOCK). Called by hcal.run.

  calibrate  per (seed, partition): H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP on the 2,000 CALIBRATION_HELDOUT
             representatives (+ T-TOKEN32 on the 2,000 CALIBRATION_TRAIN_MATCHED representatives for the 4 fixed
             diagnostic partitions); per seed: U H-GLOBAL-TEMP and H-CLASS-TEMP (hcal.calib). Units cal__s{k}__<p>,
             calU__s{k}.
  utility    per seed: inner utility (qpc.utility.release_inner_utility on INNER_SELECTION), decision preservation,
             finiteness, and calibration / train-matched / fitting-row losses of EVERY registered release (289 code
             variants + 3 U variants) -> util__s{k}; then Ucal*, the utility table and the registered audit plan
             (hcal.select) -> <PRIVATE_CACHE>/hcal_v1/run/audit_plan.json. No attack is fitted before the plan exists.
  audit      per audited (seed, partition): fresh bank on ATTACK_FIT_NEW + admitted legacy banks -> common record
             (hcal.bank) -> fam__s{k}__<p>, com__s{k}__<p>
  compose    per seed: continuous U's composed record -> com__s{k}__SRC_U
  controls   real-data controls (hcal.controls) -> run/controls.json
  replay     seed-0 refit replay of every frozen common-bank winner that the assessment will need -> run/replay.json
Labels are read only through hcal.data's allowlist; no assessment row is indexed here.
"""
from __future__ import annotations

import json
import time

import numpy as np

from hcal import ids as I

TASKS = ("income", "occupation")
K_OF = {1: 2, 2: 6}


def _R():
    from hcal import run as R
    return R


def cal_name(k, p):
    return f"cal__s{k}__{I.safe(p)}"


def calu_name(k):
    return f"calU__s{k}"


def util_name(k):
    return f"util__s{k}"


def fam_name(k, p):
    return f"fam__s{k}__{I.safe(p)}"


def com_name(k, key):
    return f"com__s{k}__{I.safe(key)}"


def cal_labels(D, role):
    """(row positions, {task: labels}) of the calibration representatives of `role` (allowlist procedure
    "calibration")."""
    from hcal import data as HD
    rows, out = None, {}
    for t in TASKS:
        r, y = HD.task_labels(D, t, "calibration", role)
        if rows is not None and not np.array_equal(r, rows):
            raise AssertionError("calibration rows differ across tasks")
        rows, out[t] = r, y
    return rows, out


# ------------------------------------------------------------------ calibrate
def calibrate_jobs():
    return [(k, p) for k in I.SEEDS for p in I.partitions()] + [(k, None) for k in I.SEEDS]


def stage_calibrate(D, shard_spec=None):
    from hcal import calib as C
    R = _R()
    h_rows, h_y = cal_labels(D, "CALIBRATION_HELDOUT")
    t_rows, t_y = cal_labels(D, "CALIBRATION_TRAIN_MATCHED")
    for k, p in R.shard(calibrate_jobs(), shard_spec):
        name = cal_name(k, p) if p is not None else calu_name(k)
        if R.done(name):
            continue
        t0, c0 = time.time(), time.process_time()
        if p is None:
            T = R.teacher(k)
            arrays, recs = {}, {}
            for fam in C.TEMP_FAMILIES:
                for i in (1, 2):
                    P = np.asarray(T[f"p{i}"], dtype=np.float64)
                    d = np.asarray(T[f"d{i}"], dtype=np.int64)
                    tab = C.fit_u_temp(P[h_rows], h_y[TASKS[i - 1]], fam, d[h_rows])
                    q = C.apply_u(P, tab, d)
                    arrays[f"{fam}|p{i}"] = q
                    recs[f"{fam}|r{i}"] = C._jsonable({x: tab[x] for x in ("kind", "target", "K", "alpha", "alphas",
                                                                          "status", "status_counts", "parameter_count",
                                                                          "certificate", "calibration_scores",
                                                                          "n_cal", "content_sha256")})
            r = {"schema": "hcal-calU-v1", "seed": k, "tables": recs, "rows": "CALIBRATION_HELDOUT representatives",
                 "n_cal": int(h_rows.size), "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
            R.save(name, {"u.npz": arrays}, r)
            continue
        b = R.bank(k, p)
        arrays, tables, summ = {}, {}, {}
        fams = [f for f in I.decoders_of(p) if f in I.NEW_DECODERS + (I.DIAG_DECODER,)]
        for fam in fams:
            rows, ys = (t_rows, t_y) if fam == I.DIAG_DECODER else (h_rows, h_y)
            tabs = C.calibrate_partition(b, rows, ys, fam)
            for i in (1, 2):
                arrays[f"{fam}|q{i}"] = tabs[i]["q"]
                tables[f"{fam}|r{i}"] = C.decoder_table(fam, i, tabs[i])
                summ[f"{fam}|r{i}"] = {x: tables[f"{fam}|r{i}"][x] for x in
                                       ("alpha", "alphas", "status_counts", "parameter_count", "calibration_scores",
                                        "content_sha256")}
                summ[f"{fam}|r{i}"]["tokens"] = int(tabs[i]["T"])
                summ[f"{fam}|r{i}"]["occupied_by_calibration_rows"] = int((np.asarray(tabs[i]["n_cal"]) > 0).sum())
                cs = tables[f"{fam}|r{i}"]["certificate"]
                summ[f"{fam}|r{i}"]["certificate_summary"] = cs.get("summary", cs)
        r = {"schema": "hcal-cal-v1", "seed": k, "partition": p, "families": fams, "tables": summ,
             "rows": {"H": "CALIBRATION_HELDOUT representatives (2,000)",
                      "T": "CALIBRATION_TRAIN_MATCHED representatives (2,000)"},
             "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        R.save(name, {"tables.npz": arrays, "tables.json": tables}, r)


# ------------------------------------------------------------------ release arrays
def tables_of(k, p):
    """(bank arrays, {decoder: (table_1, table_2)}) of partition p at seed k, in registered decoder order."""
    R = _R()
    b = R.bank(k, p)
    z = R.npz(cal_name(k, p), "tables.npz")
    out = {}
    for dec in I.decoders_of(p):
        if dec in ("D0", "MEAN"):
            out[dec] = (b["q01"], b["q02"])
        elif dec == "D1":
            out[dec] = (b["qD11"], b["qD12"])
        else:
            out[dec] = (z[f"{dec}|q1"], z[f"{dec}|q2"])
    return b, out


def release_arrays(k, rid):
    """({1: tok, 2: tok} or None, {1: q rows, 2: q rows}, {1: hard, 2: hard}) over ALL rows of one release."""
    R = _R()
    if rid.startswith(I.U_ID):
        T = R.teacher(k)
        _, fam = I.parse_release(rid)
        if fam == "identity":
            q = {i: np.asarray(T[f"p{i}"], dtype=np.float64) for i in (1, 2)}
        else:
            z = R.npz(calu_name(k), "u.npz")
            q = {i: z[f"{fam}|p{i}"] for i in (1, 2)}
        return None, q, {i: np.asarray(T[f"d{i}"], dtype=np.int64) for i in (1, 2)}
    p, dec = I.parse_release(rid)
    b, tabs = tables_of(k, p)
    tok = {i: np.asarray(b[f"tok{i}"], dtype=np.int64) for i in (1, 2)}
    return tok, {i: np.asarray(tabs[dec][i - 1], dtype=np.float64)[tok[i]] for i in (1, 2)}, \
        {i: np.asarray(b[f"hard{i}"], dtype=np.int64) for i in (1, 2)}


# ------------------------------------------------------------------ utility
def _metrics_rows(D, P, hard, rows, y, i):
    from qpc import utility as UT
    return UT.metrics(np.asarray(P)[rows], np.asarray(hard)[rows], y, K_OF[i], UT.constant_class(D, i - 1),
                      UT.constant_prior(D, i - 1))


def release_utility(D, k, rid, T, roles):
    from qpc import utility as UT
    tok, q, hard = release_arrays(k, rid)
    finite = all(bool(np.all(np.isfinite(q[i]))) for i in (1, 2))
    pres = {}
    for i in (1, 2):
        same = bool(np.array_equal(hard[i], np.asarray(T[f"d{i}"])))
        amax = bool(np.array_equal(np.asarray(q[i]).argmax(1), hard[i]))
        pres[str(i)] = same and amax
    inner = UT.release_inner_utility({1: q[1], 2: q[2]}, {1: hard[1], 2: hard[2]}, D)
    rec = {"inner": inner, "preserved": pres, "finite": finite}
    for tag, (rows, ys) in roles.items():
        rec[tag] = {TASKS[i - 1]: _metrics_rows(D, q[i], hard[i], rows, ys[TASKS[i - 1]], i) for i in (1, 2)}
    return rec


def stage_utility(D, shard_spec=None):
    from hcal import select as SEL
    from qpc import utility as UT
    R = _R()
    from hcal import data as HD
    roles = {"cal": cal_labels(D, "CALIBRATION_HELDOUT"), "tm": cal_labels(D, "CALIBRATION_TRAIN_MATCHED")}
    fr = {}
    for t in TASKS:
        r, y = UT.task_labels(D, TASKS.index(t), "fitting", "OSF_DEFENSE_FIT")
        fr["rows"], fr[t] = r, y
    roles["fit"] = (fr["rows"], {t: fr[t] for t in TASKS})
    _ = HD
    for k in I.SEEDS:
        name = util_name(k)
        if R.done(name):
            continue
        t0, c0 = time.time(), time.process_time()
        T = R.teacher(k)
        rel = {}
        for rid in I.code_release_ids() + I.u_release_ids():
            rel[rid] = release_utility(D, k, rid, T, roles)
        refs = {}
        for ref in I.REFERENCES:
            unit = (f"aud__tea__s{k}__RAW-J_b0.3" if ref.startswith("SRC|") else f"aud__ref__s{k}__{ref.split('|')[1]}")
            rr = json.loads((I.ADM_UNITS / unit / "record.json").read_text())
            refs[ref] = {"inner_utility_lra": rr.get("utility"), "recovery_lra": {
                w: (rr.get("recovery") or {}).get("auc", {}).get(w) for w in ("v1", "v2", "pair")},
                "status": "DESCRIPTIVE_ONLY (admitted lra record; no refit)"}
        r = {"schema": "hcal-util-v1", "seed": k, "releases": rel, "references": refs,
             "rows": {"inner": "INNER_SELECTION (all rows)", "cal": "CALIBRATION_HELDOUT representatives",
                      "tm": "CALIBRATION_TRAIN_MATCHED representatives", "fit": "OSF_DEFENSE_FIT (all rows)"},
             "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        R.save(name, {}, r)
    util = {k: R.rec(util_name(k))["releases"] for k in I.SEEDS}
    uc = SEL.ucal_star(util)
    if uc["status"] != "SELECTED":
        raise SystemExit(f"TECHNICAL FAILURE: Ucal* not selectable: {uc}")
    table = SEL.utility_table(util, uc["family"])
    plan = SEL.audit_plan(table)
    plan.update({"ucal_star": uc, "schema": "hcal-audit-plan-v1",
                 "written_before_any_new_attack_fit": True,
                 "eligible_releases": [r for r, v in table.items() if v["eligible_all_seeds"]],
                 "eligible_u0_only_releases": [r for r, v in table.items() if v["eligible_u0_all_seeds"]
                                               and not v["eligible_all_seeds"]]})
    p = R.RUN / "audit_plan.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("audited") != plan["audited"] or old["ucal_star"]["family"] != uc["family"]:
            raise SystemExit("REFUSED: an audit plan with different content already exists")
    else:
        p.write_text(json.dumps(R._finite(plan), indent=1, allow_nan=False) + "\n")
    (R.RUN / "utility_table.json").write_text(json.dumps(R._finite(table), indent=1, allow_nan=False) + "\n")
    R.event("audit plan", audited=len(plan["audited"]), skipped=len(plan["skipped"]), ucal=uc["family"])


# ------------------------------------------------------------------ audit (fresh banks + common records)
def attack_rows(D):
    from hcal import data as HD
    fit_rows, _ = HD.sex_labels(D, "attack", "ATTACK_FIT_NEW")
    sel_rows, _ = HD.sex_labels(D, "attack", "INNER_SELECTION")
    return fit_rows, sel_rows


def fresh_views_of(k, p, D):
    from hcal import bank as BK
    b, tabs = tables_of(k, p)
    return b, BK.fresh_views(b, tabs, D, expected=I.decoders_of(p), meta={"partition": p, "seed": k})


def stage_audit(D, shard_spec=None):
    from hcal import bank as BK
    R = _R()
    plan = json.loads((R.RUN / "audit_plan.json").read_text())
    fit_rows, sel_rows = attack_rows(D)
    y = np.asarray(D["sex"])
    hashes = BK.role_hashes(D)
    jobs = [(k, p) for p in plan["audited"] for k in I.SEEDS]
    for k, p in R.shard(jobs, shard_spec):
        fn, cn = fam_name(k, p), com_name(k, p)
        if R.done(cn):
            continue
        if not R.done(fn):
            t0, c0 = time.time(), time.process_time()
            b, views = fresh_views_of(k, p, D)
            rec, arrays = BK.fresh_family(views, D, fit_rows, sel_rows, y=y)
            trec, tarr = BK.fresh_token_family(BK.token_views(b, D), D, fit_rows, sel_rows, y=y)
            pub = BK.public_fresh(rec)
            pub.update({"seed": k, "partition": p, "decoder_variants": list(I.decoders_of(p)),
                        "token_diagnostic": BK.public_fresh(trec), "fit_cpu_s": time.process_time() - c0,
                        "fit_wall_s": time.time() - t0})
            R.save(fn, {"inner_preds.npz": BK.fresh_unit_arrays(rec, arrays),
                        "token_preds.npz": BK.fresh_unit_arrays(trec, tarr, prefix="token")}, pub)
        fresh = R.rec(fn)
        legacy = BK.load_legacy(k, p)
        com = BK.common_record(legacy, fresh, partition=p, seed=k, hashes=hashes, fresh_unit=fn)
        receipt = BK.check_common(BK.assign_variants(k, p, com), com)
        R.save(cn, {}, {"schema": "hcal-com-unit-v1", "seed": k, "partition": p, "recovery": com,
                        "variants_receipt": receipt, "fresh_unit": fn,
                        "legacy_units": [x[0] for x in legacy]})


def stage_compose(D, shard_spec=None):
    from hcal import bank as BK
    R = _R()
    plan = json.loads((R.RUN / "audit_plan.json").read_text())
    hashes = BK.role_hashes(D)
    for k in I.SEEDS:
        cn = com_name(k, "SRC|U")
        if R.done(cn):
            continue
        u, codes = BK.load_u(k)
        entry = BK.u_entry(u, codes)
        items = [(p, R.rec(fam_name(k, p)), fam_name(k, p)) for p in I.partitions() if p in plan["audited"]]
        rec = BK.compose_u(entry, items, seed=k, hashes=hashes, complete=False)
        R.save(cn, {}, {"schema": "hcal-com-unit-v1", "seed": k, "partition": "SRC|U", "recovery": rec,
                        "composed_fresh_partitions": [x[0] for x in items],
                        "not_composed_fresh": [p for p in I.partitions() if p not in plan["audited"]],
                        "note": "partitions skipped as PREDECLARED_UTILITY_INELIGIBLE have no fresh bank; their "
                                "legacy banks are inside the admitted lra SRC|U composed bank"})


# ------------------------------------------------------------------ refit views and replay
def views_for(detail, k, D):
    """Views on which a frozen winner was fitted: fresh (hcal fresh complete view), legacy / lra_code (the original
    lra complete code view of its release), lra_source (U's source interface)."""
    from lra import audit as LA
    from lra import baselines as LB
    kind = detail["kind"]
    if kind == "fresh":
        p = I.parse_release(detail["release"])[0] if detail.get("release") else None
        if p is None:
            p = _R().rec(detail["unit"])["partition"]
        return fresh_views_of(k, p, D)[1]
    if kind in ("legacy", "lra_code"):
        unit = I.lra_unit(k, detail["release"])
        z = np.load(I.ADM_UNITS / unit / "release.npz", allow_pickle=False)
        return LA.policy_views({x: z[x] for x in z.files}, D)
    if kind == "lra_source":
        return LB.source_view_sets(_R().teacher(k), D)["interface"]
    raise ValueError(f"unknown bank kind {kind}")


def refit_winner(k, key, w, crit, D, score_rows=None):
    """(P_sel, P_score) of the frozen (view w, criterion crit) winner of partition key (or SRC|U) at seed k; REFUSES
    unless the INNER refits reproduce the stored predictions bitwise (hcal.bank.refit_selected)."""
    from hcal import bank as BK
    from hcal import data as HD
    R = _R()
    rec = R.rec(com_name(k, key))["recovery"]
    det = rec["winner_detail"][w][crit]
    stored = BK.stored_predictions(det, units_dir=I.UNITS if det["kind"] == "fresh" else I.ADM_UNITS)
    spec = BK.refit_spec(rec, w, crit, stored)
    fit_role = BK.FIT_ROLE_OF[det["kind"]]
    fit_rows, _ = HD.sex_labels(D, "attack", fit_role)
    _, sel_rows = attack_rows(D)
    sc = np.zeros(0, dtype=np.int64) if score_rows is None else np.asarray(score_rows, dtype=np.int64)
    return BK.refit_selected(spec, lambda: views_for(det, k, D), D, np.asarray(D["sex"]), fit_rows, sel_rows, sc)


def replay_keys(sel):
    keys = ["SRC|U"] + list(I.DIAGNOSTIC_PARTITIONS)
    for role in ("T*", "P*"):
        rid = (sel.get(role) or {}).get("release")
        if rid and not rid.startswith(I.U_ID):
            keys.append(I.parse_release(rid)[0])
    return list(dict.fromkeys(keys))


def stage_replay(D, shard_spec=None):
    R = _R()
    sel = json.loads((R.RUN / "selection.json").read_text())
    out = {"schema": "hcal-replay-v1", "keys": replay_keys(sel), "rows": []}
    for k in I.SEEDS:
        for key in out["keys"]:
            for w in ("v1", "v2", "pair"):
                for crit in ("auc", "ce"):
                    t0 = time.time()
                    det = R.rec(com_name(k, key))["recovery"]["winner_detail"][w][crit]
                    try:
                        refit_winner(k, key, w, crit, D)
                        ok, err = True, None
                    except Exception as e:                                     # noqa: BLE001
                        ok, err = False, f"{type(e).__name__}: {e}"
                    out["rows"].append({"seed": k, "key": key, "view": w, "crit": crit, "bank": det["bank"],
                                        "kind": det["kind"], "attacker": det.get("attacker"), "ok": ok, "error": err,
                                        "wall_s": round(time.time() - t0, 2)})
    out["all_ok"] = all(r["ok"] for r in out["rows"])
    (R.RUN / "replay.json").write_text(json.dumps(R._finite(out), indent=1, allow_nan=False) + "\n")
    R.event("replay", all_ok=out["all_ok"], n=len(out["rows"]))


# ------------------------------------------------------------------ real-data controls
def control_tables(k, p):
    return tables_of(k, p)[1]


def stage_controls(D, shard_spec=None):
    from hcal import controls as CT
    R = _R()
    plan = CT.control_plan()
    jobs = CT.control_jobs(plan)
    pdir = R.RUN / "controls_parts"
    pdir.mkdir(parents=True, exist_ok=True)
    idx = list(range(len(jobs)))
    for j in R.shard(idx, shard_spec):
        out = pdir / f"part_{j:02d}.json"
        if out.exists():
            continue
        t0, c0 = time.time(), time.process_time()
        part = CT.run_controls(D, plan, load_bank_fn=R.bank, tables_fn=control_tables, jobs=[jobs[j]])[0]
        body = {"job": list(jobs[j]), "part": part, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(R._finite(body), allow_nan=False))
        tmp.rename(out)
        R.event("control part", job=list(jobs[j]))
    have = sorted(pdir.glob("part_*.json"))
    if len(have) == len(jobs):
        parts = [json.loads(p.read_text())["part"] for p in have]
        summ = CT.summarise_controls(parts, plan)
        (R.RUN / "controls.json").write_text(json.dumps(R._finite(summ), indent=1, allow_nan=False) + "\n")
        from hcal import select as SEL
        R.event("controls", all_ok=SEL.controls_all_ok(summ))
