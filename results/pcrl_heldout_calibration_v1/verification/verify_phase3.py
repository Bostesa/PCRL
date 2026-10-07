"""Role E phase 3: independent verification of the single locked assessment of the hcal study (prompt section 13).

Reads only serialized artifacts (EVALUATION_LOCK.json, the oprob__ / oatt__ assessment units, run/inference.json,
PRIMARY_ENDPOINTS.csv, the admitted bank / teacher / release units and the inner run units) and the SEALED D
(lra.data.load()). Assessment labels come ONLY from the oprob__ units (hash-complete; lock_sha256 = the pushed lock);
no study gate is used or bypassed. Every formula is re-derived here (own code; role E helpers from verify_inner.py).

  L  LOCK_AND_UNITS     EVALUATION_LOCK.json byte-identical on origin; every assessment unit hash-complete, carrying
                        the lock's sha256 and commit; the expected unit set exactly; the locked code files and every
                        unit file the lock binds unchanged.
  a  ASSESSMENT_ROLE    13,936 rows / 13,929 exact-record groups; the row-id hash vs the lock's assessment_role; every
                        unit's assess_row_id / assess_unit equal D's; identical label / constant arrays across units;
                        const_class = own OSF_DEFENSE_FIT majority class.
  b  RELEASES           every oprob unit's prob / hard on the assessment rows = table[tok] rebuilt from the admitted
                        bank + stored calibration tables (U: teacher p / stored U rows), bitwise; argmax(prob) = hard =
                        the teacher decision.
  c  ATTACK_REFITS      registered sample: every frozen winner of SRC|U, of P*'s and of T*'s partitions at seed 0, plus
                        every cell-reader winner (all seeds / keys); own sklearn factories with the pinned
                        hyperparameters (LR / MLP / HGB, DA = own Canon + LR / MLP; CC = own Dirichlet cell readers);
                        own views (legacy code view, U interface view, fresh view + hygiene; fingerprints vs the lock);
                        refit at attacker seeds 0-2 on the winner's own fit rows; INNER predictions must equal the
                        stored ones and assessment predictions the oatt__ arrays, bitwise.
  d  PRIMARY_FAMILY     the 23 slots by an own paired exact-record-group multinomial bootstrap (B = 1999,
                        default_rng(20261011), one multinomial(n_groups, uniform) per replicate in sequence, groups =
                        np.unique(assess_unit)); weighted AUC / means; seed means; SE ddof 1; z = NormalDist().inv_cdf(
                        1 - 0.05 / 46); clause outcomes; D readings; criteria; FittingRole / ParameterSharing; label.
  e  SUPPLEMENTARY      the fixed supplementary contrasts (nominal 95%, z = NormalDist().inv_cdf(0.975)).

GATES. Assessment units (oprob__ / oatt__) and inference outputs are opened only with --go (given after role A's
message that the assessment and inference are done) and only when EVALUATION_LOCK.json is byte-identical on origin.
--inner-only runs check c against the stored INNER predictions only (no assessment unit opened). Labels read from D:
SEX of AUDIT_FIT / ATTACK_FIT_NEW (attacker fitting rows) and INNER_SELECTION (cell-pair fallback rule; stored-INNER
comparison), task labels of OSF_DEFENSE_FIT (constant class only); all recorded. No hcal module is imported (checked).

    ~/PCRL/.venv/bin/python -P <WT>/hcal/sema.py --label E:phase3 -- env OMP_NUM_THREADS=1 \\
        PCRL_HCAL_PRIVATE_CACHE=$HOME/PCRL_eval_cache_private/hcal_v1 ~/PCRL/.venv/bin/python \\
        <WT>/results/pcrl_heldout_calibration_v1/verification/verify_phase3.py --go
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("verify_inner_e", HERE.parent / "verify_inner.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)                          # role E's own helpers (no hcal import)

OUT = HERE.parent / "E_PHASE3.json"
LOCK_REL = f"{V.REL}/EVALUATION_LOCK.json"
LOCK = V.WT / LOCK_REL
B_REPS, BOOT_SEED, CHUNK = 1999, 20261011, 250
Z_PRIMARY = NormalDist().inv_cdf(1 - 0.05 / (2 * 23))
Z_SUP = NormalDist().inv_cdf(0.975)
TOL = 1e-12
Z_TOL = 1e-13
DIAG_P = "U|JOINT|i8o64|l0.1"
SUP_PARTS = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", DIAG_P)
SUP_CONTRASTS = (("fitting_role", "T-TOKEN32", "H-TOKEN32"), ("parameter_sharing", "H-TOKEN32", "H-GLOBAL-TEMP"),
                 ("class_vs_global", "H-GLOBAL-TEMP", "H-CLASS-TEMP"), ("class_vs_token", "H-TOKEN32", "H-CLASS-TEMP"),
                 ("original_d0_vs_global", "D0", "H-GLOBAL-TEMP"), ("original_d1_vs_token", "D1", "H-TOKEN32"))
LABELS_ORDER = ("ENGINEERING_BLOCKED_NOT_RUN", "INPUTS_UNAVAILABLE_NOT_RUN", "INCOMPLETE_OR_INVALID",
                "NO_ELIGIBLE_COMPETITIVE_NOMINEE", "CALIBRATED_PRIVATE_RELEASE_DEVELOPMENT_CRITERION_ESTABLISHED",
                "ORIGINAL_REQUIREMENTS_MET_CALIBRATED_REFERENCE_NOT_ESTABLISHED",
                "NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED")
warnings.filterwarnings("ignore")


def sha_file(p):
    return V.sha_file(p)


# ------------------------------------------------------------------ gated assessment-unit access
class Gate:
    def __init__(self, go):
        self.go = go
        self.lock_ok, self.lock_why = V.on_origin(LOCK_REL)
        self.lock_sha = sha_file(LOCK) if LOCK.exists() else None

    def open_ok(self):
        return self.go and self.lock_ok


def load_assess_unit(name, gate):
    if not name.startswith(("oprob__", "oatt__")):
        raise ValueError(name)
    if not gate.open_ok():
        raise PermissionError("REFUSED: assessment units are opened only with --go after the pushed EVALUATION_LOCK")
    d = V.UNITS / name
    c = d / "COMPLETE.json"
    if not c.exists():
        raise V.Missing(f"{name}: no COMPLETE.json")
    files = json.loads(c.read_text())["files"]
    for f, h in files.items():
        got = sha_file(d / f)
        V.INPUT_HASHES[V.pub(d / f)] = got
        if got != h:
            raise V.Corrupt(f"{name}/{f}: sha256 differs from COMPLETE.json")
    rec = json.loads((d / "record.json").read_text())
    z = np.load(d / "preds.npz", allow_pickle=False)
    return rec, {k: z[k] for k in z.files}


def prob_name(k, rid):
    return f"oprob__s{k}__{V.safe(rid)}"


def att_name(k, key):
    return f"oatt__s{k}__{V.safe(key)}"


def partition_key(rid):
    return V.U_ID if rid in V.U_RIDS else V.RID_INFO[rid][0]


# ------------------------------------------------------------------ L: lock and units
def check_lock(L, gate):
    chk = V.Check("L_LOCK_AND_UNITS", "EVALUATION_LOCK pushed; assessment units hash-complete and bound to the lock")
    chk.ok(gate.lock_ok, f"EVALUATION_LOCK: {gate.lock_why}")
    commit = V.git("log", "-1", "--format=%H", "--", LOCK_REL).stdout.strip()
    chk.info["lock_commit"] = commit
    chk.info["lock_sha256"] = gate.lock_sha
    commit_time = int(V.git("log", "-1", "--format=%ct", commit).stdout.strip() or 0)
    chk.info["lock_commit_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(commit_time))
    chk.ok(f"origin/{V.BRANCH}" in V.git("branch", "-r", "--contains", commit).stdout.split(),
           "lock commit not on origin")
    changed = [f for f, h in (L.get("locked_code_files") or {}).items()
               if not (V.WT / f).exists() or sha_file(V.WT / f) != h]
    chk.ok(not changed, f"locked code changed after EVALUATION_LOCK: {changed[:10]}")
    chk.info["locked_code_files"] = len(L.get("locked_code_files") or {})
    for k, s in L["seeds"].items():
        for unit, files in s["unit_file_sha256"].items():
            try:
                got = json.loads((V.UNITS / unit / "COMPLETE.json").read_text())["files"]
            except FileNotFoundError:
                chk.bad(f"bound unit {unit} missing")
                continue
            chk.same("bound unit files", got, files, unit)
    if not gate.open_ok():
        chk.pend("assessment units not opened (needs --go and the pushed lock)")
        return chk.done()
    want = {prob_name(k, r) for k in V.SEEDS for r in L["scored_releases"]} | \
           {att_name(k, x) for k in V.SEEDS for x in L["attack_keys"]}
    have = {p.name for p in V.UNITS.iterdir() if p.name.startswith(("oprob__", "oatt__")) and p.is_dir()}
    chk.same("assessment unit set", sorted(have), sorted(want))
    for nm in sorted(want & have):
        try:
            rec, _ = load_assess_unit(nm, gate)
        except (V.Missing, V.Corrupt) as e:
            chk.bad(str(e))
            continue
        chk.same("unit lock_sha256", rec.get("lock_sha256"), gate.lock_sha, nm)
        chk.same("unit lock_commit", rec.get("lock_commit"), commit, nm)
        mt = (V.UNITS / nm / "COMPLETE.json").stat().st_mtime
        chk.ok(mt > commit_time, f"{nm}: written ({mt:.0f}) before the lock commit ({commit_time})")
        chk.info["earliest_unit_minus_lock_commit_s"] = min(chk.info.get("earliest_unit_minus_lock_commit_s", 1e18),
                                                            round(mt - commit_time, 1))
        if nm.startswith("oatt__"):
            k = int(nm.split("__")[1][1:])
            key = rec.get("key")
            for w in V.VIEWS:
                for crit in V.CRITS:
                    d = L["seeds"][str(k)]["winners"][key][w][crit]
                    got = (rec.get("winners") or {}).get(f"{crit}_{w}") or {}
                    chk.same("oatt winner", (got.get("bank"), got.get("kind"), got.get("attacker"), got.get("view")),
                             (d["bank"], d["kind"], d.get("attacker"), d.get("view")), f"{nm} {crit}/{w}")
    return chk.done()


# ------------------------------------------------------------------ a + b: role, base arrays, releases
BASE = {}


def check_role_and_releases(D, L, gate, LB, roles):
    ca = V.Check("a_ASSESSMENT_ROLE", "assessment rows / groups / hash vs the lock; base arrays identical across units")
    cb = V.Check("b_RELEASES", "oprob prob / hard = table[tok] (bank + stored tables; U rows), bitwise; decisions")
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
    rid = np.asarray(D["row_id"])
    grp = np.asarray(D["unit"])
    mine = {"rows": int(a.size), "groups": int(np.unique(grp[a]).size), "row_id_sha256": V.sha_arr(rid[a])}
    ca.same("assessment role vs lock", mine, L["assessment_role"])
    ca.same("rows / groups", (mine["rows"], mine["groups"]), (13936, 13929))
    const = {}
    for j, t in enumerate(V.TASKS):
        _, yf = LB.task(t, "OSF_DEFENSE_FIT", "constant class only")
        const[j] = int(np.argmax(np.bincount(yf, minlength=V.KS[j + 1])))
    if not gate.open_ok():
        ca.pend("assessment units not opened")
        cb.pend("assessment units not opened")
        return ca.done(), cb.done()
    ref = None
    for k in V.SEEDS:
        T = V.teacher(k)
        for r in L["scored_releases"]:
            nm = prob_name(k, r)
            try:
                _, z = load_assess_unit(nm, gate)
            except (V.Missing, V.Corrupt) as e:
                cb.bad(str(e))
                continue
            if ref is None:
                ref = z
                ca.ok(np.array_equal(z["assess_row_id"], rid[a]), "assess_row_id != D's assessment rows")
                ca.ok(np.array_equal(z["assess_unit"], grp[a]), "assess_unit != D's exact-record groups")
                ca.ok(all(int(np.min(z[x])) >= 0 for x in ("sex", "y_income", "y_occ")), "negative labels in a unit")
                ca.same("const_class", [int(x) for x in z["const_class"]], [const[0], const[1]])
            for x in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ", "const_class"):
                ca.ok(np.array_equal(z[x], ref[x]), f"{nm}: {x} differs across units")
            if r in V.U_RIDS:
                fam = V.U_FAM[r]
                if fam == "identity":
                    q = {i: np.asarray(T[f"p{i}"], dtype=np.float64) for i in (1, 2)}
                else:
                    _, uz = V.calu(k)
                    q = {i: np.asarray(uz[f"{fam}|p{i}"], dtype=np.float64) for i in (1, 2)}
                hard = {i: np.asarray(T[f"d{i}"], dtype=np.int64) for i in (1, 2)}
            else:
                p, d = V.RID_INFO[r]
                b = V.bank(k, p)
                if d in ("D0", "MEAN"):
                    tab = (b["q01"], b["q02"])
                elif d == "D1":
                    tab = (b["qD11"], b["qD12"])
                else:
                    _, tz, _ = V.cal_tables(k, p)
                    tab = (tz[f"{d}|q1"], tz[f"{d}|q2"])
                q = {i: np.asarray(tab[i - 1], dtype=np.float64)[b[f"tok{i}"]] for i in (1, 2)}
                hard = {i: np.asarray(b[f"hard{i}"], dtype=np.int64) for i in (1, 2)}
            for i in (1, 2):
                cb.ok(np.array_equal(z[f"prob{i}"], q[i][a]), f"{nm}: prob{i} != rebuilt release rows (bitwise)")
                cb.ok(np.array_equal(z[f"hard{i}"], hard[i][a]), f"{nm}: hard{i} != released decisions")
                cb.ok(np.array_equal(z[f"hard{i}"], np.asarray(T[f"d{i}"])[a]), f"{nm}: hard{i} != teacher d")
                cb.ok(np.array_equal(np.argmax(z[f"prob{i}"], 1), z[f"hard{i}"]), f"{nm}: argmax(prob{i}) != hard")
                cb.ok(bool(np.all(np.isfinite(z[f"prob{i}"]))), f"{nm}: non-finite prob{i}")
    if ref is not None:
        BASE.update(ref)
    ca.info["const_class"] = const
    return ca.done(), cb.done()


# ------------------------------------------------------------------ c: attacker refits (own factories / views)
class Canon:
    """Float64 canonicalisation (pinned rule): centre, SVD on the fit rows, keep s > max(n, d) eps scale, whiten."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        scale = max(float(s[0]) if s.size else 0.0, float(np.linalg.norm(X, 2)))
        tol = 1.0 * max(n, d) * float(np.finfo(np.float64).eps) * scale
        keep = s > tol
        self.rank = int(keep.sum())
        self.W = (Vt[keep].T / s[keep]) * math.sqrt(max(n - 1, 1))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) @ self.W


class DAReader:
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        self.c = Canon().fit(X)
        Z = self.c.transform(X)
        if self.c.rank == 0:
            self.m, Z = DummyClassifier(strategy="prior"), np.zeros((len(Z), 1))
        elif self.kind == "lr":
            self.m = LogisticRegression(C=1.0, max_iter=3000)
        else:
            self.m = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m.fit(Z, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        Z = self.c.transform(X)
        return self.m.predict_proba(Z if self.c.rank else np.zeros((len(Z), 1)))


def factory(att, seed):
    """Own construction of the pinned FINAL slate member `att` (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if att.startswith("LR_C"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(att[4:]), max_iter=3000))
    if att.startswith("MLP_"):
        h = tuple(int(x) for x in att[4:].split("x"))
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                             early_stopping=True, validation_fraction=0.1,
                                                             n_iter_no_change=15, random_state=seed))
    if att.startswith("HGB_"):
        _, lr, lv = att.split("_")
        return HistGradientBoostingClassifier(learning_rate=float(lr), max_leaf_nodes=int(lv), max_iter=200,
                                              early_stopping=False, random_state=seed)
    if att == "DA_LR":
        return DAReader("lr", seed)
    if att == "DA_MLP":
        return DAReader("mlp", seed)
    raise ValueError(f"not a slate member: {att}")


def p1(model, X):
    P = model.predict_proba(X)
    cl = [int(c) for c in getattr(model, "classes_", range(P.shape[1]))]
    return np.zeros(len(X)) if 1 not in cl else np.asarray(P[:, cl.index(1)], dtype=np.float64)


def cc_tab(keys, y):
    u, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv.ravel(), minlength=len(u)).astype(np.float64)
    n1 = np.bincount(inv.ravel(), weights=(np.asarray(y) == 1).astype(np.float64), minlength=len(u))
    return u, n, n1


def cc_look(tab, keys, alpha, prior):
    u, n, n1 = tab
    pos = np.clip(np.searchsorted(u, keys), 0, max(len(u) - 1, 0))
    seen = (u[pos] == keys) if len(u) else np.zeros(len(keys), bool)
    p = np.full(len(keys), prior, dtype=np.float64)
    p[seen] = (n1[pos[seen]] + alpha * prior) / (n[pos[seen]] + alpha)
    return p, seen


def cc_predict(att, toks, view, y, fit, sel, score, ys_sel):
    """Own exact-identity cell readers: (P_sel, P_score, fallback rule or None)."""
    alpha = float(att.split("alpha")[1])
    prior = float(np.mean(np.asarray(y)[fit] == 1))
    if not att.startswith("CCpair"):
        t = np.asarray(toks[view], dtype=np.int64)
        tab = cc_tab(t[fit], y[fit])
        return cc_look(tab, t[sel], alpha, prior)[0], cc_look(tab, t[score], alpha, prior)[0], None
    t1, t2 = np.asarray(toks["v1"], dtype=np.int64), np.asarray(toks["v2"], dtype=np.int64)
    keys = t1 * (int(max(t2.max(initial=0), 0)) + 1) + t2
    tab = cc_tab(keys[fit], y[fit])
    tabs = {1: cc_tab(t1[fit], y[fit]), 2: cc_tab(t2[fit], y[fit])}

    def cand(rows):
        return {"local_1": cc_look(tabs[1], t1[rows], alpha, prior)[0],
                "local_2": cc_look(tabs[2], t2[rows], alpha, prior)[0], "prior": np.full(len(rows), prior)}
    ps, seen_s = cc_look(tab, keys[sel], alpha, prior)
    cs = cand(sel)
    un = ~seen_s
    if un.sum() == 0:
        rule = "prior"
    else:
        ces = {r: V.ce_fixed(ys_sel[un], cs[r][un]) for r in ("local_1", "local_2", "prior")}
        rule = min(("local_1", "local_2", "prior"), key=lambda r: (ces[r], ("local_1", "local_2", "prior").index(r)))
    ps = ps.copy()
    ps[un] = cs[rule][un]
    pc, seen_c = cc_look(tab, keys[score], alpha, prior)
    cc = cand(score)
    pc = pc.copy()
    pc[~seen_c] = cc[rule][~seen_c]
    return ps, pc, rule


def code_view(z, D):
    X, T = {}, {}
    for i in (1, 2):
        tok = np.asarray(z[f"tok{i}"], dtype=np.int64)
        a = int(np.asarray(z[f"alpha{i}"]).ravel()[0])
        X[f"v{i}"] = np.hstack([V.onehot(V.canonical_cols(tok, D, a)[tok], a), np.asarray(z[f"q{i}"], dtype=np.float64),
                                V.onehot(np.asarray(z[f"hard{i}"], dtype=np.int64), V.KS[i])])
        T[f"v{i}"] = tok
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return X, T


def u_interface_view(t):
    n = len(t["row_id"])
    X = {f"v{i}": np.hstack([np.asarray(t[f"c{i}"], dtype=np.float64).reshape(n, -1),
                             np.asarray(t[f"p{i}"], dtype=np.float64).reshape(n, -1)]) for i in (1, 2)}
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return X, None


def fresh_view(k, p, D, roles):
    b = V.bank(k, p)
    _, tz, _ = V.cal_tables(k, p)
    tabs = {d: ((b["q01"], b["q02"]) if d in ("D0", "MEAN") else (b["qD11"], b["qD12"]) if d == "D1" else
                (tz[f"{d}|q1"], tz[f"{d}|q2"])) for d in V.decoders(p)}
    F = np.sort(np.asarray(roles["ATTACK_FIT_NEW"], dtype=np.int64))
    X, T = {}, {}
    for i in (1, 2):
        tok = np.asarray(b[f"tok{i}"], dtype=np.int64)
        a = int(b[f"alpha{i}"])
        blocks = [V.onehot(V.canonical_cols(tok, D, a)[tok], a)]
        blocks += [np.asarray(tabs[d][i - 1], dtype=np.float64)[tok] for d in V.decoders(p)]
        blocks.append(V.onehot(np.asarray(b[f"hard{i}"], dtype=np.int64), V.KS[i]))
        X[f"v{i}"], _ = V.hygiene(np.hstack(blocks), F)
        T[f"v{i}"] = tok
    X["pair"], _ = V.hygiene(np.hstack([X["v1"], X["v2"]]), F)
    return X, T


_VIEWS = {}


def views_of(det, k, D, roles):
    kind = det["kind"]
    if kind == "fresh":
        p = V.unit_record(det["unit"])[0]["partition"]
        key = ("fresh", k, p)
        if key not in _VIEWS:
            _VIEWS[key] = fresh_view(k, p, D, roles)
    elif kind in ("legacy", "lra_code"):
        unit = V.lra_release_unit(k, det["release"])
        key = ("code", unit)
        if key not in _VIEWS:
            f = V.load_unit(unit, V.ADM_UNITS)
            _VIEWS[key] = code_view(V.npz(f["release.npz"]), D)
    elif kind == "lra_source":
        key = ("src", k)
        if key not in _VIEWS:
            _VIEWS[key] = u_interface_view(V.teacher(k))
    else:
        raise ValueError(kind)
    return _VIEWS[key]


def stored_inner(det):
    root = V.UNITS if det["kind"] == "fresh" else V.ADM_UNITS
    f = V.load_unit(det["unit"], root)
    return np.asarray(V.npz(f["inner_preds.npz"])[det["stored_key"]], dtype=np.float64)


def registered_sample(L):
    """(seed, key, view, crit) of the registered refit sample (module docstring)."""
    keys0 = {V.U_ID, V.RID_INFO[L["resolved"]["P*"]][0] if L["resolved"].get("P*") in V.RID_INFO else None,
             V.RID_INFO[L["resolved"]["T*"]][0] if L["resolved"].get("T*") in V.RID_INFO else None}
    if L["resolved"].get("T*") in V.U_RIDS:
        keys0.add(V.U_ID)
    out = []
    for k, s in L["seeds"].items():
        for key, wv in s["winners"].items():
            for w, cd in wv.items():
                for crit, det in cd.items():
                    if (int(k) == 0 and key in keys0) or str(det.get("attacker", "")).startswith("CC"):
                        out.append((int(k), key, w, crit))
    return sorted(out)


def check_refits(D, L, gate, LB, roles, inner_only):
    chk = V.Check("c_ATTACK_REFITS", "registered sample of frozen winners refit with own factories / views, bitwise")
    sel_idx = np.asarray(D["idx"]["INNER_SELECTION"], dtype=np.int64)
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
    fit_rows = {"AUDIT_FIT": np.asarray(D["idx"]["AUDIT_FIT"], dtype=np.int64),
                "ATTACK_FIT_NEW": np.sort(np.asarray(roles["ATTACK_FIT_NEW"], dtype=np.int64))}
    _, ys_sel = LB.get("sex", "INNER_SELECTION", "cell-pair fallback rule; INNER refit comparison")
    sample = registered_sample(L)
    chk.info["sample"] = [f"s{k} {key} {w}/{c}" for k, key, w, c in sample]
    cache = {}
    oatt = {}
    for k, key, w, crit in sample:
        det = L["seeds"][str(k)]["winners"][key][w][crit]
        where = f"s{k} {key} {w}/{crit} {det['kind']}:{det.get('attacker')}"
        role = "ATTACK_FIT_NEW" if det["kind"] == "fresh" else "AUDIT_FIT"
        chk.same("fit role", det.get("fit_role"), role, where)
        fit = fit_rows[role]
        _, yfit = LB.get("sex", role, "attacker fitting rows (refit of frozen winners)")
        y = np.zeros(len(D["row_id"]), dtype=np.int64)
        y[fit] = yfit
        y[sel_idx] = ys_sel
        chk.same("fit rows sha", det.get("fit_row_id_sha256"), V.sha_arrays(np.asarray(D["row_id"])[fit]), where)
        chk.same("INNER rows sha", det.get("sel_row_id_sha256"), V.sha_arrays(np.asarray(D["row_id"])[sel_idx]), where)
        X, T = views_of(det, k, D, roles)
        if det.get("view_fingerprint"):
            chk.same("view fingerprint", det["view_fingerprint"], V.sha_arrays(X["v1"], X["v2"], X["pair"]), where)
        att, view = det["attacker"], det["view"]
        ck = (det["kind"], det.get("unit"), det.get("release"), k, view, att)
        if ck not in cache:
            if att.startswith("CC"):
                ps, pc, rule = cc_predict(att, T, view, y, fit, sel_idx, a, ys_sel)
                cache[ck] = ([ps] * 3, [pc] * 3, rule)
            else:
                Xv = np.asarray(X[view], dtype=np.float64)
                P_sel, P_sc = [], []
                for sd in (0, 1, 2):
                    m = factory(att, sd).fit(Xv[fit], y[fit])
                    P_sel.append(p1(m, Xv[sel_idx]))
                    P_sc.append(p1(m, Xv[a]))
                cache[ck] = (P_sel, P_sc, None)
        P_sel, P_sc, rule = cache[ck]
        stored = stored_inner(det)
        chk.ok(stored.shape == (3, sel_idx.size), f"{where}: stored INNER shape {stored.shape}")
        for sd in range(3):
            chk.ok(np.array_equal(P_sel[sd], stored[sd]), f"{where}: own INNER refit != stored (attacker seed {sd})")
        if inner_only or not gate.open_ok():
            continue
        if (k, key) not in oatt:
            try:
                oatt[(k, key)] = load_assess_unit(att_name(k, key), gate)[1]
            except (V.Missing, V.Corrupt) as e:
                chk.bad(str(e))
                continue
        Z = np.asarray(oatt[(k, key)][f"P_{crit}_{w}"], dtype=np.float64)
        chk.ok(Z.shape == (3, a.size, 2), f"{where}: oatt shape {Z.shape}")
        for sd in range(3):
            chk.ok(np.array_equal(Z[sd, :, 1], P_sc[sd]) and np.array_equal(Z[sd, :, 0], 1.0 - P_sc[sd]),
                   f"{where}: own assessment refit != oatt (attacker seed {sd}); max |diff| "
                   f"{float(np.max(np.abs(Z[sd, :, 1] - P_sc[sd]))):.3e}")
    if inner_only or not gate.open_ok():
        chk.pend("assessment predictions not compared (--inner-only or units not opened)")
    return chk.done()


# ------------------------------------------------------------------ d + e: bootstrap inference
def wauc_fn(score, pos):
    """Weighted fixed-orientation AUC (ties count 1/2) as a function of row weights WT (n, c)."""
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    s = np.asarray(score, dtype=np.float64)[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos_o = np.asarray(pos, bool)[order]

    def f(WT):
        Wo = WT[order]
        gp = np.add.reduceat(Wo * pos_o[:, None], starts, axis=0)
        gn = np.add.reduceat(Wo * (~pos_o)[:, None], starts, axis=0)
        below = np.cumsum(gn, 0) - gn
        num = (gp * (below + 0.5 * gn)).sum(0)
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)
    return f


def wmean_fn(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return lambda WT: (vec @ WT) / WT.sum(0)


def clause(side, target, pt, lo, hi):
    if not all(x is not None and math.isfinite(x) for x in (pt, lo, hi)):
        return "INVALID"
    if side == "lower>":
        if lo > target:
            return "PASS"
        if hi < target:
            return "MEASURED_VIOLATION"
        return "NOT_ESTABLISHED_PRECISION" if pt > target else "NOT_ESTABLISHED_POINT"
    if hi < target:
        return "PASS"
    if lo > target:
        return "MEASURED_VIOLATION"
    return "NOT_ESTABLISHED_PRECISION" if pt < target else "NOT_ESTABLISHED_POINT"


def diag(pt, lo, hi):
    if not all(x is not None and math.isfinite(x) for x in (pt, lo, hi)):
        return "INVALID"
    return "SUPPORTS_POSITIVE" if lo > 0 else ("SUPPORTS_NEGATIVE" if hi < 0 else "UNRESOLVED")


def diag_summary(vals):
    vals = list(vals)
    if any(v == "INVALID" for v in vals):
        return "INVALID"
    for v, s in (("SUPPORTS_POSITIVE", "ALL_FOUR_SUPPORT_POSITIVE"), ("SUPPORTS_NEGATIVE", "ALL_FOUR_SUPPORT_NEGATIVE"),
                 ("UNRESOLVED", "ALL_FOUR_UNRESOLVED")):
        if all(x == v for x in vals):
            return s
    return "MIXED"


ORIG = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"]
CAL = ["P12", "P13", "P14", "P15"]


def criteria(out, nominee_valid, technical_valid, comparator_valid):
    if not technical_valid:
        return "INVALID_TECHNICAL", "INVALID_TECHNICAL"
    if not comparator_valid:
        return "INVALID_NO_COMPARATOR", "INVALID_NO_COMPARATOR"
    if not nominee_valid:
        return "NOT_TESTED_NO_NOMINEE", "NOT_TESTED_NO_NOMINEE"
    o = "PASS" if all(out.get(i) == "PASS" for i in ORIG) else "NOT_ESTABLISHED"
    c = "PASS" if o == "PASS" and all(out.get(i) == "PASS" for i in CAL) else "NOT_ESTABLISHED"
    return o, c


def overall(eng, inputs, tv, nstate, cstate, out):
    """PROTOCOL section 7 precedence (re-derived)."""
    if not eng:
        return LABELS_ORDER[0]
    if not inputs:
        return LABELS_ORDER[1]
    if not tv or "TECHNICAL_FAILURE" in (nstate, cstate) or cstate != "ELIGIBLE":
        return LABELS_ORDER[2]
    if nstate != "ELIGIBLE":
        return LABELS_ORDER[3]
    if any(out.get(i) == "INVALID" for i in ORIG + CAL):
        return LABELS_ORDER[2]
    o, c = criteria(out, True, True, True)
    if c == "PASS":
        return LABELS_ORDER[4]
    return LABELS_ORDER[5] if o == "PASS" else LABELS_ORDER[6]


def slot_defs():
    """The 23 registered slots (PRIMARY_FAMILY.json), re-stated: (id, kind, extras)."""
    s = [("P01", "coalition", {"side": "lower>", "target": 0.02})]
    s += [("P02", "local", {"view": "v1", "side": "upper<", "target": 0.01}),
          ("P03", "local", {"view": "v2", "side": "upper<", "target": 0.01})]
    for base, kind, tgt, side in ((4, "acc", -0.01, "lower>"), (6, "ll", 0.01, "upper<"), (8, "br", 0.005, "upper<"),
                                  (10, "ret", 0.0, "lower>")):
        s += [(f"P{base + j:02d}", kind, {"task": j, "side": side, "target": tgt, "ref": "U0"}) for j in (0, 1)]
    for base, kind, tgt in ((12, "ll", 0.01), (14, "br", 0.005)):
        s += [(f"P{base + j:02d}", kind, {"task": j, "side": "upper<", "target": tgt, "ref": "Ucal*"}) for j in (0, 1)]
    order = ((0, "ll"), (1, "ll"), (0, "br"), (1, "br"))
    s += [(f"D{n + 1:02d}", "diag", {"task": j, "loss": ls, "a": "T-TOKEN32", "b": "H-TOKEN32"})
          for n, (j, ls) in enumerate(order)]
    s += [(f"D{n + 5:02d}", "diag", {"task": j, "loss": ls, "a": "H-TOKEN32", "b": "H-GLOBAL-TEMP"})
          for n, (j, ls) in enumerate(order)]
    return s


def check_inference(L, gate):
    cd = V.Check("d_PRIMARY_FAMILY", "23 slots by an own paired exact-record-group bootstrap; outcomes; label")
    ce = V.Check("e_SUPPLEMENTARY", "fixed supplementary contrasts (nominal 95%)")
    if not gate.open_ok() or not BASE:
        cd.pend("assessment units / inference not opened")
        ce.pend("assessment units / inference not opened")
        return cd.done(), ce.done()
    try:
        inf = V.read_json(V.RUN / "inference.json")
    except V.Missing as e:
        cd.pend(str(e))
        ce.pend(str(e))
        return cd.done(), ce.done()
    cd.diff("z (registered expression)", L["family"]["z"], Z_PRIMARY, Z_TOL)
    cd.diff("z (inference)", inf["z"], Z_PRIMARY, Z_TOL)
    cd.same("B / seed", (inf["B"], inf["seed"], L["family"]["B"], L["family"]["bootstrap_seed"]),
            (B_REPS, BOOT_SEED, B_REPS, BOOT_SEED))
    sex = np.asarray(BASE["sex"])
    y = {0: np.asarray(BASE["y_income"], dtype=np.int64), 1: np.asarray(BASE["y_occ"], dtype=np.int64)}
    const = {j: int(BASE["const_class"][j]) for j in (0, 1)}
    units = np.asarray(BASE["assess_unit"])
    uniq, idx = np.unique(units, return_inverse=True)
    cd.same("n_assessment / n_groups", (inf["n_assessment"], inf["n_groups"]), (int(units.size), int(uniq.size)))
    R = L["resolved"]
    nom, tref, ucal, u0 = R.get("P*"), R.get("T*"), R.get("Ucal*"), R.get("U0")
    probs, atts = {}, {}

    def P(k, rid):
        if (k, rid) not in probs:
            probs[(k, rid)] = load_assess_unit(prob_name(k, rid), gate)[1]
        return probs[(k, rid)]

    def A(k, key):
        if (k, key) not in atts:
            atts[(k, key)] = load_assess_unit(att_name(k, key), gate)[1]
        return atts[(k, key)]
    stats = {}                                      # name -> f(WT)

    def base(name, fn):
        if name not in stats:
            stats[name] = fn()
        return name

    def rec_ids(k, rid, view):
        key = partition_key(rid)
        Z = np.asarray(A(k, key)[f"P_auc_{view}"], dtype=np.float64)
        return [base(f"auc#{k}#{key}#{view}#{s}", lambda s=s: wauc_fn(Z[s, :, 1], sex == 1)) for s in range(3)]

    def acc_id(k, rid, j):
        return base(f"acc#{k}#{rid}#{j}", lambda: wmean_fn(P(k, rid)[f"hard{j + 1}"] == y[j]))

    def loss_id(k, rid, j, kind):
        def mk():
            pr = np.asarray(P(k, rid)[f"prob{j + 1}"], dtype=np.float64)
            n = len(y[j])
            if kind == "ll":
                v = -np.log(np.clip(pr[np.arange(n), y[j]], 1e-12, 1.0))
            else:
                oh = np.zeros_like(pr)
                oh[np.arange(n), y[j]] = 1.0
                v = ((pr - oh) ** 2).sum(1)
            return wmean_fn(v)
        return base(f"{kind}#{k}#{rid}#{j}", mk)

    def const_id(j):
        return base(f"const#{j}", lambda: wmean_fn(y[j] == const[j]))
    # endpoint recipes: list of (per-seed combiner) -> evaluated per WT
    slots = slot_defs()
    recipe = {}
    for sid, kind, ex in slots:
        if kind == "diag":
            a_, b_ = V.rid_of(DIAG_P, ex["a"]), V.rid_of(DIAG_P, ex["b"])
            recipe[sid] = [("diff", [loss_id(k, a_, ex["task"], ex["loss"]), loss_id(k, b_, ex["task"], ex["loss"])])
                           for k in V.SEEDS]
            continue
        if nom is None or tref is None or ucal is None:
            recipe[sid] = None
            continue
        per = []
        for k in V.SEEDS:
            if kind == "coalition":
                per.append(("rdiff", [rec_ids(k, tref, "pair"), rec_ids(k, nom, "pair")]))
            elif kind == "local":
                per.append(("rdiff", [rec_ids(k, nom, ex["view"]), rec_ids(k, tref, ex["view"])]))
            elif kind == "acc":
                per.append(("diff", [acc_id(k, nom, ex["task"]), acc_id(k, u0, ex["task"])]))
            elif kind in ("ll", "br"):
                ref = u0 if ex["ref"] == "U0" else ucal
                per.append(("diff", [loss_id(k, nom, ex["task"], kind), loss_id(k, ref, ex["task"], kind)]))
            else:
                per.append(("lin", [acc_id(k, nom, ex["task"]), acc_id(k, u0, ex["task"]), const_id(ex["task"])]))
        recipe[sid] = per
    sup_recipe = {}
    scored = set(L["scored_releases"])
    for dp in SUP_PARTS:
        for name, ad, bd in SUP_CONTRASTS:
            if ad not in V.decoders(dp) or bd not in V.decoders(dp):
                continue
            a_, b_ = V.rid_of(dp, ad), V.rid_of(dp, bd)
            if a_ not in scored or b_ not in scored:
                continue
            for j in (0, 1):
                for kind in ("ll", "br"):
                    sup_recipe[(dp, name, kind, j)] = [("diff", [loss_id(k, a_, j, kind), loss_id(k, b_, j, kind)])
                                                       for k in V.SEEDS]

    def endpoint(per, v):
        vals = []
        for op, parts in per:
            if op == "rdiff":
                x = np.mean(np.stack([v[s] for s in parts[0]]), axis=0)
                yv = np.mean(np.stack([v[s] for s in parts[1]]), axis=0)
                vals.append(x - yv)
            elif op == "diff":
                vals.append(v[parts[0]] - v[parts[1]])
            else:
                vals.append(v[parts[0]] - 0.8 * v[parts[1]] - 0.2 * v[parts[2]])
        return np.mean(np.stack(vals), axis=0)

    def evaluate(WT):
        v = {nm: np.asarray(f(WT), dtype=np.float64) for nm, f in stats.items()}
        e = {sid: endpoint(per, v) for sid, per in recipe.items() if per is not None}
        s = {key: endpoint(per, v) for key, per in sup_recipe.items()}
        return e, s
    pts, spts = evaluate(np.ones((units.size, 1)))
    reps = {sid: [] for sid in pts}
    sreps = {key: [] for key in spts}
    rng = np.random.default_rng(BOOT_SEED)
    pvec = np.full(uniq.size, 1.0 / uniq.size)
    done = 0
    while done < B_REPS:
        c = min(CHUNK, B_REPS - done)
        counts = np.stack([rng.multinomial(uniq.size, pvec) for _ in range(c)], axis=1)
        e, s = evaluate(counts[idx].astype(np.float64))
        for sid in reps:
            reps[sid].append(e[sid])
        for key in sreps:
            sreps[key].append(s[key])
        done += c
    stored = {r["id"]: r for r in inf["primary"]}
    mine_out = {}
    for sid, kind, ex in slots:
        st = stored.get(sid, {})
        if recipe[sid] is None:
            mine_out[sid] = "NOT_SCORED_NO_NOMINEE"
            cd.same("decision", st.get("decision"), "NOT_SCORED_NO_NOMINEE", sid)
            continue
        r = np.concatenate(reps[sid])
        pt = float(pts[sid][0])
        if (~np.isfinite(r)).any() or not math.isfinite(pt):
            mine_out[sid] = "INVALID"
        else:
            se = float(np.std(r, ddof=1))
            lo, hi = pt - Z_PRIMARY * se, pt + Z_PRIMARY * se
            mine_out[sid] = diag(pt, lo, hi) if kind == "diag" else clause(ex["side"], ex["target"], pt, lo, hi)
            cd.diff("point", st.get("point"), pt, TOL, sid)
            cd.diff("se", st.get("se"), se, TOL, sid)
            cd.diff("lower", st.get("lower"), lo, TOL, sid)
            cd.diff("upper", st.get("upper"), hi, TOL, sid)
            if kind != "diag":
                band = min(abs(lo - ex["target"]), abs(hi - ex["target"]), abs(pt - ex["target"]))
            else:
                band = min(abs(lo), abs(hi))
            if band <= 1e-9:
                cd.notes.append(f"{sid}: a bound or the point lies within 1e-9 of its threshold ({band:.2e})")
            cd.info.setdefault("slots", {})[sid] = {"point": pt, "se": se, "lower": lo, "upper": hi,
                                                     "decision": mine_out[sid]}
        cd.same("decision", st.get("decision"), mine_out[sid], sid)
    nstate, cstate = L["statuses"]["P*"]["state"], L["statuses"]["T*"]["state"]
    tv = bool((L.get("technical_validity") or {}).get("ok")) and not (inf.get("failures_input") or {}).get("global")
    o, c = criteria(mine_out, nstate == "ELIGIBLE", tv and "TECHNICAL_FAILURE" not in (nstate, cstate),
                    cstate == "ELIGIBLE")
    label = overall(True, True, tv, nstate, cstate, mine_out)
    fit_ids, share_ids = ["D01", "D02", "D03", "D04"], ["D05", "D06", "D07", "D08"]
    cd.same("OriginalCriterion", inf.get("OriginalCriterion"), o)
    cd.same("CalibrationMatchedCriterion", inf.get("CalibrationMatchedCriterion"), c)
    cd.same("FittingRole", (inf.get("FittingRole") or {}).get("summary"), diag_summary(mine_out[i] for i in fit_ids))
    cd.same("ParameterSharing", (inf.get("ParameterSharing") or {}).get("summary"),
            diag_summary(mine_out[i] for i in share_ids))
    cd.same("label", inf.get("label"), label)
    cd.same("technical_valid", inf.get("technical_valid"), tv)
    cd.info.update({"OriginalCriterion": o, "CalibrationMatchedCriterion": c, "label": label,
                    "FittingRole": diag_summary(mine_out[i] for i in fit_ids),
                    "ParameterSharing": diag_summary(mine_out[i] for i in share_ids)})
    # the label truth table, re-derived with the own precedence
    try:
        tt = V.read_json(V.PKG / "LABEL_TRUTH_TABLE.json")
        cd.same("truth-table precedence", tt["precedence"], list(LABELS_ORDER))
    except V.Missing as e:
        cd.pend(str(e))
    # PRIMARY_ENDPOINTS.csv agrees with inference.json decisions
    pe = V.PKG / "PRIMARY_ENDPOINTS.csv"
    if pe.exists():
        V.INPUT_HASHES[V.pub(pe)] = sha_file(pe)
        with open(pe) as fh:
            rows = {r["id"]: r for r in csv.DictReader(fh)}
        for sid in mine_out:
            cd.same("PRIMARY_ENDPOINTS.csv decision", (rows.get(sid) or {}).get("decision"), mine_out[sid], sid)
    else:
        cd.pend("PRIMARY_ENDPOINTS.csv missing")
    srows = {(r["partition"], r["contrast"], "ll" if r["loss"] == "logloss" else "br",
              {"income": 0, "occupation": 1}[r["task"]]): r for r in inf["supplementary"]["rows"]}
    ce.diff("z95", inf["supplementary"]["z"], Z_SUP, Z_TOL)
    ce.same("supplementary rows", sorted(map(str, srows)), sorted(map(str, sup_recipe)))
    for key in sup_recipe:
        r = np.concatenate(sreps[key])
        pt = float(spts[key][0])
        st = srows.get(key)
        if st is None:
            continue
        if not np.isfinite(r).all() or not math.isfinite(pt):
            ce.same("se", st.get("se"), None, str(key))
            continue
        se = float(np.std(r, ddof=1))
        lo, hi = pt - Z_SUP * se, pt + Z_SUP * se
        for nm, mv in (("point", pt), ("se", se), ("lower", lo), ("upper", hi)):
            ce.diff(nm, st.get(nm), mv, TOL, str(key))
        ce.same("reading", st.get("reading"), diag(pt, lo, hi), str(key))
    return cd.done(), ce.done()


# ------------------------------------------------------------------ main
def main():
    t0, c0 = time.time(), time.process_time()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    go, inner_only = "--go" in sys.argv, "--inner-only" in sys.argv
    sys.path.insert(0, str(V.WT))
    from lra import data as LD                       # pinned loader only (sealed D)
    D = LD.load(verify=True, unseal=False)
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
    assert D["sealed"] and all(np.all(np.asarray(D[x])[a] == -1) for x in ("sex", "y_income", "y_occupation_group"))
    sl_ok, sl_why = V.on_origin(f"{V.REL}/SCIENCE_LOCK.json")
    if not sl_ok:
        raise SystemExit(f"REFUSED: {sl_why}")
    gate = Gate(go and not inner_only)
    L = V.read_json(LOCK)
    h = V.split(D, "AUDIT_FIT", V.SALT_H)
    t = V.split(D, "OSF_DEFENSE_FIT", V.SALT_T)
    roles = {"CALIBRATION_HELDOUT": h["reps"], "ATTACK_FIT_NEW": h["rest_rows"], "CALIBRATION_TRAIN_MATCHED": t["reps"]}
    LB = V.Labels(D, roles)
    LB.get = _labels_get_with_attack(LB)
    check_lock(L, gate)
    check_role_and_releases(D, L, gate, LB, roles)
    check_refits(D, L, gate, LB, roles, inner_only)
    check_inference(L, gate)
    loaded_hcal = sorted(m for m in sys.modules if m == "hcal" or m.startswith("hcal."))
    statuses = {k: v["status"] for k, v in V.CHECKS.items()}
    if loaded_hcal or any(s == "FAIL" for s in statuses.values()):
        verdict = "FAIL"
    elif all(s == "PASS" for s in statuses.values()):
        verdict = "PASS"
    else:
        verdict = "PENDING"
    out = {"schema": "hcal-E-phase3-v1", "role": "E (independent verifier)",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script": f"{V.REL}/verification/verify_phase3.py", "script_sha256": sha_file(HERE),
           "helpers": {"verify_inner.py": sha_file(HERE.parent / "verify_inner.py")}, "argv": sys.argv[1:],
           "head": V.git("rev-parse", "HEAD").stdout.strip(),
           "evaluation_lock": {"pushed": gate.lock_ok, "reason": gate.lock_why, "sha256": gate.lock_sha},
           "assessment_units_opened": gate.open_ok(),
           "label_reads": V.LABEL_READS + ([{"label": "sex, y_income, y_occ", "role": "OSF_DEVELOPMENT_ASSESSMENT",
                                              "source": "oprob__ units (hash-complete, bound to the pushed lock)"}]
                                            if BASE else []),
           "independence": {"hcal_modules_loaded": loaded_hcal,
                            "imports": ["numpy", "sklearn (own slate construction)", "statistics.NormalDist",
                                        "lra.data (pinned loader; load() only)", "role E verify_inner.py helpers"]},
           "tolerances": {"points_se_bounds": TOL, "z": Z_TOL, "predictions": "bitwise"},
           "statuses": statuses, "checks": V.CHECKS, "inputs_sha256": dict(sorted(V.INPUT_HASHES.items())),
           "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1), "verdict": verdict}
    txt = json.dumps(V.jsafe(out), indent=1, allow_nan=False) + "\n"
    for bad in (str(Path.home()) + "/", "/Users/", "/private/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter E_PHASE3.json")
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(txt)
    tmp.rename(OUT)
    print(json.dumps({"verdict": verdict, "statuses": statuses, "lock": gate.lock_why,
                      "units_opened": gate.open_ok(), "wall_s": out["wall_s"]}, indent=1))


def _labels_get_with_attack(LB):
    """verify_inner.Labels extended with the attacker fitting roles (SEX of AUDIT_FIT / ATTACK_FIT_NEW; procedure
    'attack'); assessment and HEAD_VALIDATION stay refused."""
    orig = LB.get

    def get(key, role, purpose):
        if key == "sex" and role in ("AUDIT_FIT", "ATTACK_FIT_NEW"):
            if (key, role) in LB._cache:
                return LB._cache[(key, role)]
            rows = LB._rows(role)
            if np.intersect1d(rows, LB.assess).size or np.intersect1d(rows, LB.head).size:
                raise PermissionError(f"REFUSED: {role} rows touch the assessment or HEAD_VALIDATION")
            yv = np.asarray(LB.D["sex"], dtype=np.int64)[rows]
            if np.any(yv < 0):
                raise PermissionError(f"REFUSED: sealed labels on {role}")
            V.LABEL_READS.append({"label": key, "role": role, "rows": int(rows.size), "purpose": purpose,
                                  "row_id_sha256": V.sha_arr(np.sort(np.asarray(LB.D["row_id"])[rows]))})
            LB._cache[(key, role)] = (rows, yv)
            return rows, yv
        return orig(key, role, purpose)
    return get


if __name__ == "__main__":
    main()
