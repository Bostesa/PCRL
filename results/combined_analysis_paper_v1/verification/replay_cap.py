#!/usr/bin/env python
"""Independent verifier for the useful-head comparison (cap, combined_analysis_paper_v1).

Rebuilds roles, heads, release surfaces, attacker replays, bank / plus selections, unit resolution, recovery AUCs,
the paired group bootstrap, bounds and decisions from raw saved arrays only, and compares them with the runner's
outputs (inference.json and the endpoint CSVs). It never imports the runner's packages; an import guard on
sys.meta_path raises on any attempt (direct or indirect, e.g. via unpickling).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_analysis_paper_v1/verification/replay_cap.py

Writes results/combined_analysis_paper_v1/INDEPENDENT_VERIFICATION.json (aggregates, hashes and counts only).
"""
from __future__ import annotations

import importlib.abc
import sys

FORBIDDEN = ("cap", "odx", "oar", "stored_model_eval", "report", "pcrl")


class _ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ImportError(f"verifier import guard: forbidden package '{fullname}'")
        return None


sys.meta_path.insert(0, _ImportGuard())
_preloaded = [m for m in sys.modules if m.split(".")[0] in FORBIDDEN]
if _preloaded:
    raise SystemExit(f"forbidden modules already loaded: {_preloaded}")

import os  # noqa: E402

os.environ["OMP_NUM_THREADS"] = "1"

import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402
from sklearn.base import clone  # noqa: E402

T0 = time.time()
HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private"
CAPU = PRIV / "cap_v1" / "run" / "units"
OARR = PRIV / "oar_v1" / "run"
OARU = OARR / "units"
ODXU = PRIV / "odx_v1" / "run" / "units"
BENCH = PRIV / "bench_v1"
PKG = Path(__file__).resolve().parents[1]
WT = Path(__file__).resolve().parents[3]
INF = json.loads((PRIV / "cap_v1" / "run" / "inference.json").read_text())
LOCK = json.loads((PKG / "LOCK.json").read_text())
LOCK_COMMIT = "a0de449539f5753dc75e8927538d7105b7e19fa1"

SEEDS = (0, 1, 2)
AS = (0, 1, 2)
ARMS = ("A", "B", "F", "F0")
TAG = {"A": "A", "B": "B", "F": "F", "F0": "FZ"}
CONTRASTS = (("A", "B"), ("A", "F"), ("A", "F0"), ("B", "F"), ("F0", "F"))
NB, BSEED, CHUNK = 1999, 20261041, 400
Z_P = float(norm.ppf(1 - 0.05 / 38))
Z_S = float(norm.ppf(1 - 0.05 / 66))

checks = []


def tilde(p) -> str:
    p = Path(p)
    try:
        return "~/" + str(p.relative_to(HOME))
    except ValueError:
        return str(p)


def add(cid, status, detail, max_abs_diff=None, **extra):
    row = {"id": cid, "status": status, "detail": detail}
    if max_abs_diff is not None:
        row["max_abs_diff"] = float(max_abs_diff)
    row.update(extra)
    checks.append(row)
    print(f"[{status}] {cid}: {detail}" + (f" (max_abs_diff={max_abs_diff:.3g})" if max_abs_diff is not None else ""),
          flush=True)


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_arr(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def uid(k, arm, view, surface):
    return f"cap__s{k}__{arm}__{view}__{surface}"


def npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ================================================================================================ 1. roles
L = np.load(BENCH / "inputs" / "adult_labels.npz", allow_pickle=False)
row_id, unit_all, canon = L["row_id"], L["unit"], L["canon_key"]
split = L["split"]
role0 = L["role"].astype("<U32")
s_all = L["sex"].astype(int)
t_all = L["task_income"].astype(int)
SCORED = ("attacker_fit", "attacker_val", "assessment")
train_keys = set(canon[split == "train"].tolist())
exposed = (split == "test") & np.fromiter((k in train_keys for k in canon.tolist()), bool, len(canon))
role = role0.copy()
excl = exposed & np.isin(role0, SCORED)
role[excl] = "excluded_exposure"


def u01(salt: str, key: str) -> float:
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 2 ** 32


cert = (role == "attacker_fit") & np.fromiter((u01("oar-cert-v1|adult|", k) < 0.20 for k in canon.tolist()), bool,
                                               len(canon))
role[cert] = "cert"
IDX = {r: np.flatnonzero(role == r) for r in ("attacker_fit", "attacker_val", "assessment", "cert", "excluded_exposure")}
fI, vI, aI = IDX["attacker_fit"], IDX["attacker_val"], IDX["assessment"]
counts = {r: int(len(v)) for r, v in IDX.items()}
exp_counts = {"assessment": 5243, "attacker_fit": 6065, "attacker_val": 2235}
ok = all(counts[r] == n for r, n in exp_counts.items())
add("roles.counts", "PASS" if ok else "FAIL",
    f"rebuilt oar-roles-v1: {counts}; exposure-excluded scored rows={int(excl.sum())}; expected {exp_counts}")
ROWS_A, UNITS_A, YS_A, YT_A = row_id[aI], unit_all[aI], s_all[aI], t_all[aI]
n_groups = len(np.unique(UNITS_A))
add("roles.groups", "PASS" if n_groups == len(aI) == INF["n_assessment_groups"] == INF["n_assessment_rows"] else "FAIL",
    f"assessment groups={n_groups}, rows={len(aI)}; runner n_rows={INF['n_assessment_rows']} n_groups={INF['n_assessment_groups']}")
for r in ("attacker_fit", "attacker_val", "assessment", "cert"):
    if len(np.intersect1d(unit_all[IDX[r]], unit_all[role == "defense_fit"])):
        add(f"roles.disjoint.{r}", "FAIL", "role shares groups with defense_fit")
role_hashes = {r: sha_arr(np.sort(row_id[v]).astype(np.int64)) for r, v in IDX.items()}

# ================================================================================================ 2. heads
MAJ = int(np.argmax(np.bincount(t_all[fI])))
MINO = int(np.argmin(np.bincount(t_all[fI])))
cK = (YT_A == MAJ).astype(float)
const_acc = float(cK.mean())
HEAD = {}
lse_max = off_max = 0.0
util_rows = {}
for k in SEEDS:
    for arm in ARMS:
        p = npz(OARU / f"adult__s{k}__HEAD__{TAG[arm]}" / "preds.npz")
        Lh = p["head_outputs_all"].astype(np.float64)
        assert np.array_equal(p["row_id"], row_id), "head row order differs from labels"
        assert np.array_equal(p["assess_row_id"], ROWS_A) and np.array_equal(p["y_t"], YT_A)
        HEAD[k, arm] = Lh
        l0, l1 = Lh[:, 0], Lh[:, 1]
        lse_max = max(lse_max, float(np.max(np.abs(np.logaddexp(l0, l1)))))
        d = l1 - l0
        c = (l0 + l1) / 2
        off_max = max(off_max, float(np.max(np.abs(c + (np.logaddexp(0, d) + np.logaddexp(0, -d)) / 2))))
        P = np.exp(Lh[aI])
        yhat = P.argmax(1)
        rec = [float(np.mean(yhat[YT_A == cc] == cc)) for cc in (0, 1)]
        acc = float(np.mean(yhat == YT_A))
        util_rows[k, arm] = {"accuracy": acc, "balanced_accuracy": float(np.mean(rec)), "minority_class": MINO,
                             "minority_recall": rec[MINO],
                             "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(YT_A)), YT_A], 1e-12, 1)))),
                             "constant_accuracy": const_acc, "gain_over_constant": acc - const_acc,
                             "share_predicted_positive": float(np.mean(yhat == 1))}
add("heads.logsumexp", "PASS" if lse_max < 1e-9 else "FAIL", "max |logsumexp(row)| over 12 heads x 39,205 rows",
    lse_max)
add("heads.offset_identity", "PASS" if off_max < 1e-9 else "FAIL",
    "max |c + (softplus(d)+softplus(-d))/2| over 12 heads (offset is a deterministic function of the margin)", off_max)
add("heads.constant", "PASS" if abs(const_acc - 0.7492) < 5e-5 else "FAIL",
    f"majority task class on attacker_fit = {MAJ}; constant accuracy on assessment = {const_acc:.6f} (protocol 0.7492); minority class={MINO}")

# utility vs CSV (6-decimal formatting) and inference.json (full precision)
FIELDS = ("accuracy", "balanced_accuracy", "minority_recall", "log_loss", "constant_accuracy", "gain_over_constant",
          "share_predicted_positive")
with open(PKG / "ACTUAL_HEAD_UTILITY.csv") as fh:
    csv_util = {(int(r["seed"]), r["arm"]): r for r in csv.DictReader(fh)}
mx_csv = mx_inf = 0.0
fmt_bad = 0
inf_util = {(int(r["seed"]), r["arm"]): r for r in INF["utility"]}
for key, mine in util_rows.items():
    for fld in FIELDS:
        mx_csv = max(mx_csv, abs(mine[fld] - float(csv_util[key][fld])))
        fmt_bad += f"{mine[fld]:.6f}" != csv_util[key][fld]
        mx_inf = max(mx_inf, abs(mine[fld] - inf_util[key][fld]))
    fmt_bad += int(csv_util[key]["minority_class"]) != mine["minority_class"]
add("heads.utility_vs_inference_json", "PASS" if mx_inf < 1e-9 and len(inf_util) == 12 else "FAIL",
    "accuracy, balanced accuracy, minority recall, log loss, gain, share positive for 12 heads vs inference.json['utility']",
    mx_inf)
add("heads.utility_vs_csv", "PASS" if fmt_bad == 0 and mx_csv <= 5e-7 + 1e-12 else "FAIL",
    f"ACTUAL_HEAD_UTILITY.csv is printed to 6 decimals: {fmt_bad} formatted mismatches over 12 heads x {len(FIELDS)+1} fields "
    "(max abs diff is the rounding residual)", mx_csv)


# ================================================================================================ 3. surfaces, features, replays
def surfaces(Lh):
    Lh = np.asarray(Lh, np.float64)
    d = Lh[:, 1] - Lh[:, 0]
    c = (Lh[:, 0] + Lh[:, 1]) / 2
    m = Lh.max(1, keepdims=True)
    E = np.exp(Lh - m)
    hard = np.zeros_like(Lh)
    hard[np.arange(len(Lh)), Lh.argmax(1)] = 1.0
    return {"full": Lh, "dc": np.column_stack([d, c]), "centred": d[:, None], "prob": E / E.sum(1, keepdims=True),
            "hard": hard}


ties_d0 = sum(int(np.sum(HEAD[key][:, 1] == HEAD[key][:, 0])) for key in HEAD)
hard_eq = all(np.array_equal(HEAD[key].argmax(1), (HEAD[key][:, 1] - HEAD[key][:, 0] > 0).astype(int)) for key in HEAD)
add("surfaces.hard_rule", "PASS" if hard_eq else "FAIL",
    f"argmax decision equals 1[d>0] on all 12 heads; rows with d==0: {ties_d0}")

FWD = {}
CLEAN = {}
for k in SEEDS:
    Fz = np.load(BENCH / "inputs" / f"adult_s{k}_forward.npz", allow_pickle=False)
    assert np.array_equal(Fz["row_id"], row_id)
    FWD[k] = Fz["rep_p0"].astype(np.float64)
    CLEAN[k] = Fz["logits_income_prediction"].astype(np.float64)

SEL = {k: json.loads((OARR / "selection" / f"adult__s{k}.json").read_text()) for k in SEEDS}


def leace_numpy(k, H):
    z = npz(BENCH / "defenses" / f"adult__s{k}__income_prediction__B_sex" / "map" / "leace_map.npz")
    return H - ((H - z["mean_x"]) @ z["proj_right"].T) @ z["proj_left"].T


def onehot_cells(path):
    cells = np.load(path)
    K = int(cells.max()) + 1
    X = np.zeros((len(cells), K))
    X[np.arange(len(cells)), cells.astype(int)] = 1.0
    return X


FEAT = {}
leace_hash_ok = True
for k in SEEDS:
    FEAT[k, "A"] = FWD[k]
    md = BENCH / "defenses" / f"adult__s{k}__income_prediction__B_sex" / "map"
    for fn in ("leace_map.json", "leace_map.npz"):
        leace_hash_ok &= sha_file(md / fn) == LOCK["admitted"]["leace_maps"][tilde(md / fn)]
    FEAT[k, "B"] = leace_numpy(k, FWD[k])
    j = SEL[k]["nominee_unit_source"]
    FEAT[k, "F"] = onehot_cells(OARU / f"adult__s{k}__FAREFIT_c{j}" / "cells.npy")
    FEAT[k, "F0"] = onehot_cells(OARU / f"adult__s{k}__FAREFIT_Z" / "cells.npy")
add("features.leace_map_hashes", "PASS" if leace_hash_ok else "FAIL",
    "LEACE map json/npz sha256 equal LOCK.json admitted hashes; B features computed in numpy as "
    "x - ((x - mean_x) @ proj_right.T) @ proj_left.T (the official LeaceEraser formula), no forbidden import")
add("features.fare_nominees", "INFO",
    f"F nominee_unit_source per seed = {[SEL[k]['nominee_unit_source'] for k in SEEDS]} (protocol: 4/2/4); "
    f"one-hot widths F={[FEAT[k, 'F'].shape[1] for k in SEEDS]}, F0={[FEAT[k, 'F0'].shape[1] for k in SEEDS]}")


def full_proba(m, X, K=2):
    p = m.predict_proba(X)
    out = np.zeros((len(X), K))
    out[:, np.asarray(m.classes_).astype(int)] = p
    return out


def unit_design(k, arm, view, surface):
    """Design matrix (all 39,205 rows) of the unit uid(k, arm, view, surface) or of its alias source."""
    sf = surfaces(HEAD[k, arm])
    if view == "out":
        return sf[surface]
    if view == "feat":
        return FEAT[k, arm]
    if view == "feat+out":
        return np.hstack([FEAT[k, arm], sf[surface]])
    if view == "feat+clean":
        return np.hstack([FEAT[k, arm], CLEAN[k]])
    raise ValueError(view)


NEW = [(k, arm, "out", sn) for k in SEEDS for arm in ARMS for sn in ("dc", "centred", "prob", "hard")] + \
      [(k, arm, "feat+out", sn) for k in SEEDS for arm in ARMS for sn in ("centred", "prob", "hard")]
ALIASED = [(k, arm, v, s) for k in SEEDS for arm in ARMS for v, s in
           (("feat", "none"), ("out", "full"), ("feat+out", "full"), ("feat+clean", "full"))]
assert len(NEW) == 84 and len(ALIASED) == 48

replay = {"units": 0, "models": 0, "bitwise": 0, "max_abs": 0.0, "by_kind": {}, "fail": []}
VAL_LL = {}


def replay_unit(u, d, key, kindtag):
    k, arm, view, surface = key
    X = unit_design(k, arm, view, surface)
    p = npz(d / "preds.npz")
    rec = json.loads((d / "record.json").read_text())
    if rec.get("n_columns") is not None and rec["n_columns"] != X.shape[1]:
        replay["fail"].append(f"{u}: n_columns {rec['n_columns']} vs rebuilt {X.shape[1]}")
    worst = 0.0
    for mk, pk in [(f"NL_as{a}", f"P__NL__as{a}") for a in AS] + [("L", "P__L")]:
        m = joblib.load(d / "models" / f"{mk}.joblib")
        P = full_proba(m, X[aI])
        diff = float(np.max(np.abs(P - p[pk])))
        replay["models"] += 1
        replay["bitwise"] += bool(np.array_equal(P, p[pk]))
        worst = max(worst, diff)
    replay["units"] += 1
    replay["max_abs"] = max(replay["max_abs"], worst)
    bk = replay["by_kind"].setdefault(f"{kindtag}|{arm}", {"units": 0, "max_abs": 0.0})
    bk["units"] += 1
    bk["max_abs"] = max(bk["max_abs"], worst)
    if worst > 1e-9:
        replay["fail"].append(f"{u}: max abs {worst:.3g}")
    # validation log loss from saved validation predictions (the selection input)
    vp = npz(d / "val_preds.npz")
    vok = np.array_equal(vp["val_row_id"], row_id[vI]) and np.array_equal(vp["val_y_s"], s_all[vI])
    Pv = np.clip(vp["VAL__NL__as0"], 1e-12, 1.0)
    Pv = Pv / Pv.sum(1, keepdims=True)
    ll = float(-np.mean(np.log(Pv[np.arange(len(vI)), s_all[vI]])))
    VAL_LL[u] = (ll, rec["val_log_loss"]["NL"], vok)


for key in NEW:
    u = uid(*key)
    replay_unit(u, CAPU / u, key, f"new {key[2]}/{key[3]}")
ALIAS_SRC = {}
for key in ALIASED:
    u = uid(*key)
    al = json.loads((CAPU / u / "ALIAS.json").read_text())
    src = HOME / al["source"].removeprefix("~/")
    ALIAS_SRC[u] = src
    replay_unit(u, src, key, f"alias-source {key[2]}/{key[3]}")
B_fail = [f for f in replay["fail"] if "__B__" in f]
add("replay.saved_attackers", "PASS" if not replay["fail"] else "FAIL",
    f"reloaded NL_as0, NL_as1, NL_as2 and L for all 84 new units and all 48 alias-source units "
    f"({replay['units']} units, {replay['models']} models) and predicted on independently rebuilt assessment "
    f"surfaces (out: surface; features+output: hstack[arm features, surface]; features-only; features+clean logits); "
    f"bitwise-equal {replay['bitwise']}/{replay['models']}; failures: {replay['fail'][:10]}",
    replay["max_abs"], by_kind=replay["by_kind"])
add("replay.B_features", "PASS" if not B_fail else "FAIL",
    "B-arm units replayed with numpy LEACE (no forbidden import needed); included in replay.saved_attackers")
add("replay.cell_conditional", "INFO",
    "CC.joblib (cell-conditional NLDA, descriptive, in no family) not reloaded: its class lives in a forbidden package")
vmax = max(abs(a - b) for a, b, _ in VAL_LL.values())
vrows = all(ok for _, _, ok in VAL_LL.values())
add("selection.val_log_loss_from_val_preds", "PASS" if vmax < 1e-9 and vrows else "FAIL",
    f"recorded val_log_loss.NL vs -mean log of saved VAL__NL__as0 (clip 1e-12, renormalised) on rebuilt attacker_val "
    f"rows, {len(VAL_LL)} units; val rows/labels match rebuilt roles: {vrows}", vmax)

# --- refit one output-only unit from scratch
ru = uid(1, "B", "out", "centred")
rrec = json.loads((CAPU / ru / "record.json").read_text())
m0 = joblib.load(CAPU / ru / "models" / "NL_as0.joblib")
est = m0[-1] if hasattr(m0, "steps") else m0
prm = est.get_params()
hp_ok = all((list(prm[h]) if isinstance(prm[h], tuple) else prm[h]) == v for h, v in rrec["nl_selected"].items())
X = unit_design(1, "B", "out", "centred")
t1 = time.process_time()
mr = clone(m0)
mr.fit(X[fI], s_all[fI])
Pr = full_proba(mr, X[aI])
Ps = npz(CAPU / ru / "preds.npz")["P__NL__as0"]


def auc_mw(score, pos):
    r = rankdata(score)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(P, y):
    return float(np.mean([auc_mw(P[:, c], y == c) for c in (0, 1)]))


auc_r, auc_s = macro_auc(Pr, YS_A), macro_auc(Ps, YS_A)
Pv = full_proba(mr, X[vI])
vll = float(-np.mean(np.log(np.clip(Pv, 1e-12, 1)[np.arange(len(vI)), s_all[vI]])))
add("replay.refit_from_scratch", "INFO",
    f"{ru}: recorded NL family {rrec['nl_family']} {rrec['nl_selected']} (saved model params agree: {hp_ok}); "
    f"refit on {len(fI)} attacker_fit rows in {time.process_time() - t1:.1f}s CPU; macro AUC refit={auc_r:.6f} "
    f"saved={auc_s:.6f}; max abs prob diff={float(np.max(np.abs(Pr - Ps))):.3g}; bitwise={np.array_equal(Pr, Ps)}; "
    f"val log loss refit={vll:.10f} recorded={rrec['val_log_loss']['NL']:.10f}", abs(auc_r - auc_s))

# ================================================================================================ 4. selections


def unit_path(u):
    p = CAPU / u
    return p if p.exists() else OARU / u


def rec_of(u):
    return json.loads((unit_path(u) / "record.json").read_text())


BANK_CANDS = {("out", "iobank"): ("centred", "prob"), ("out", "fullbank"): ("full", "dc", "centred", "prob"),
              ("feat+out", "iobank"): ("centred", "prob"), ("feat+out", "fullbank"): ("full", "centred", "prob")}
bank_bad, bank_ties, nb = [], [], 0
for k in SEEDS:
    for arm in ARMS:
        for (view, b), cs in BANK_CANDS.items():
            bu = uid(k, arm, view, b)
            r = rec_of(bu)
            cands = [uid(k, arm, view, c) for c in cs]
            vals = [float(rec_of(c)["val_log_loss"]["NL"]) for c in cands]
            best = min(range(len(cands)), key=lambda i: (vals[i], i))
            nb += 1
            if sum(v == vals[best] for v in vals) > 1:
                bank_ties.append(bu)
            if r.get("kind") != "bank" or r["bank_selected"] != cands[best] or \
                    list(r["candidates_attacker_val_log_loss"]) != cands or \
                    [r["candidates_attacker_val_log_loss"][c] for c in cands] != vals:
                bank_bad.append(bu)
add("selection.banks", "PASS" if not bank_bad and nb == 48 else "FAIL",
    f"{nb} banks recomputed (argmin val_log_loss.NL, ties to earlier candidate); mismatches: {bank_bad}; "
    f"banks with an exact tie at the minimum: {len(bank_ties)} {bank_ties}")


def select_plus(cands):
    best = None
    for name, ll in cands.items():
        if ll is None or not np.isfinite(ll):
            continue
        if best is None or ll < cands[best]:
            best = name
    return best


plus_bad, plus_ties, plus_n, plus_sel = [], [], 0, {}
for k in SEEDS:
    for arm in ARMS:
        j = SEL[k]["nominee_unit_source"]
        oar_rep = f"adult__s{k}__Fc{j}__rep" if arm == "F" else f"adult__s{k}__{TAG[arm]}__rep"
        units = [(uid(k, arm, "feat+out", sn), uid(k, arm, "feat", "none"), uid(k, arm, "out", sn), "cap")
                 for sn in ("centred", "prob", "hard")]
        units.append((uid(k, arm, "feat+out", "full"), oar_rep, f"adult__s{k}__O_head{TAG[arm]}", "oar"))
        units.append((uid(k, arm, "feat+clean", "full"), oar_rep, f"adult__s{k}__O_full", "oar"))
        for u, rep_u, out_u, world in units:
            r = rec_of(u)
            cand = {"GBT/MLP": float(r["val_log_loss"]["NL"]),
                    "ignore_rep": float(rec_of(out_u)["val_log_loss"]["NL"]),
                    "ignore_out": float(rec_of(rep_u)["val_log_loss"]["NL"])}
            sel = select_plus(cand)
            src = {"ignore_rep": out_u, "ignore_out": rep_u}.get(sel)
            ps = r.get("plus_selection") or {}
            plus_n += 1
            plus_sel[u] = sel
            vals = list(cand.values())
            if vals.count(min(vals)) > 1:
                plus_ties.append(f"{u} ({[n for n, v in cand.items() if v == min(vals)]} -> {sel})")
            if ps.get("selected") != sel or ps.get("alias_source") != src or ps.get("candidates_attacker_val_log_loss") != cand:
                plus_bad.append(u)
from collections import Counter  # noqa: E402

add("selection.plus_rule", "PASS" if not plus_bad else "FAIL",
    f"{plus_n} features+output units (36 new, 12 aliased features+own full, 12 aliased features+clean) recomputed from "
    f"the three candidate validation log losses (strict <, ties to earlier listed GBT/MLP, ignore_rep, ignore_out); "
    f"mismatches: {plus_bad}; selections: {dict(Counter(plus_sel.values()))}; exact ties at the minimum: {plus_ties}")

# alias integrity
alias_bad = []
for u, src in ALIAS_SRC.items():
    al = json.loads((CAPU / u / "ALIAS.json").read_text())
    if sha_file(CAPU / u / "record.json") != sha_file(src / "record.json") or \
            al["source_COMPLETE_sha256"] != sha_file(src / "COMPLETE.json"):
        alias_bad.append(u)
add("selection.alias_integrity", "PASS" if not alias_bad else "FAIL",
    f"48 alias records byte-identical to their source record and source COMPLETE.json sha256 as recorded; bad: {alias_bad}")


def resolve(u):
    chain = [u]
    while True:
        d = unit_path(u)
        r = json.loads((d / "record.json").read_text())
        if r.get("kind") == "bank":
            u = r["bank_selected"]
            chain.append(u)
            continue
        src = (r.get("plus_selection") or {}).get("alias_source")
        if src:
            u = src
            chain.append(u)
            continue
        if (d / "ALIAS.json").exists():
            return HOME / json.loads((d / "ALIAS.json").read_text())["source"].removeprefix("~/"), chain
        return d, chain


VS = [("out", "centred"), ("out", "hard"), ("out", "full"), ("feat+out", "full"), ("feat+out", "centred"),
      ("feat+out", "hard"), ("feat", "none"), ("out", "fullbank"), ("out", "iobank"), ("feat+clean", "full"),
      ("out", "dc"), ("out", "prob"), ("feat+out", "prob"), ("feat+out", "iobank"), ("feat+out", "fullbank")]
RES = {}
for k in SEEDS:
    for arm in ARMS:
        for v, s in VS:
            RES[uid(k, arm, v, s)] = resolve(uid(k, arm, v, s))
res_bad = [u for u, rr in INF["resolution"].items() if tilde(RES[u][0]) != rr["scored_dir"] or RES[u][1] != rr["chain"]]
missing = [u for u in INF["resolution"] if u not in RES]
add("selection.resolution", "PASS" if not res_bad and not missing and len(INF["resolution"]) == 132 else "FAIL",
    f"{len(INF['resolution'])} runner resolutions compared (scored_dir and chain); mismatches: {res_bad}; "
    f"{len(set(map(str, (RES[u][0] for u in RES))))} distinct scored directories over {len(RES)} uids "
    f"(incl. 48 descriptive-level uids not in the runner's resolution table)")

# integrity of every scored directory and every cap unit
integ_bad = []
for d in sorted({rr[0] for rr in RES.values()} | {CAPU / u for u in os.listdir(CAPU)}):
    c = json.loads((d / "COMPLETE.json").read_text())
    for fn, h in c["files"].items():
        if not (d / fn).exists() or sha_file(d / fn) != h:
            integ_bad.append(f"{d.name}/{fn}")
add("integrity.complete_hashes", "PASS" if not integ_bad else "FAIL",
    f"every file in COMPLETE.json of all cap units and all scored directories hash-verifies; bad: {integ_bad[:10]}")

# ================================================================================================ 5. recovery
PRED = {}
row_bad = []
for u, (d, _) in RES.items():
    if d in PRED:
        continue
    p = npz(d / "preds.npz")
    if not (np.array_equal(p["assess_row_id"], ROWS_A) and np.array_equal(p["assess_unit"], UNITS_A)
            and np.array_equal(p["y_s"], YS_A)):
        row_bad.append(d.name)
    PRED[d] = p
for key in NEW:
    p = npz(CAPU / uid(*key) / "preds.npz")
    if not (np.array_equal(p["assess_row_id"], ROWS_A) and np.array_equal(p["assess_unit"], UNITS_A)):
        row_bad.append(uid(*key))
add("roles.assessment_rows_in_units", "PASS" if not row_bad else "FAIL",
    f"assess_row_id / assess_unit / y_s of all {len(PRED)} scored directories and all 84 new units equal the rebuilt "
    f"assessment rows (and the 12 head units' assess_row_id / y_t); mismatches: {row_bad}")

PT = {}       # unit-level point recovery (mean over attacker seeds of macro AUC)
for u, (d, _) in RES.items():
    PT[u] = float(np.mean([macro_auc(PRED[d][f"P__NL__as{a}"], YS_A) for a in AS]))
ps_diff, ps_n = 0.0, 0
for key, val in INF["per_seed"].items():
    parts = key.split("|")
    if parts[0] == "R":
        _, arm, v, s, sk = parts
        mine = PT[uid(int(sk[1:]), arm, v, s)]
    else:
        _, arm, sk = parts
        k = int(sk[1:])
        mine = float(np.mean(HEAD[k, arm][aI].argmax(1) == YT_A))
    ps_diff = max(ps_diff, abs(mine - val))
    ps_n += 1
add("recovery.per_seed", "PASS" if ps_diff < 1e-9 and ps_n == 192 else "FAIL",
    f"{ps_n} per-seed points (180 recovery uids: NL macro AUC over sex classes, mean over as0-as2; 12 head accuracies) "
    "vs inference.json['per_seed']", ps_diff)
cls_gap = max(abs(auc_mw(PRED[d]["P__NL__as0"][:, 0], YS_A == 0) - auc_mw(PRED[d]["P__NL__as0"][:, 1], YS_A == 1))
              for d in PRED)
add("recovery.class_symmetry", "INFO", "max |AUC(class 0) - AUC(class 1)| at as0 over scored dirs (binary: should be ~0)",
    cls_gap)

# ================================================================================================ 6. bootstrap


class WAUC:
    def __init__(self, score, pos):
        o = np.argsort(score, kind="stable")
        ss = score[o]
        self.o = o
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        self.pos = pos[o].astype(float)[:, None]
        self.neg = 1.0 - self.pos

    def __call__(self, Wc):
        Ws = Wc[self.o]
        gp = np.add.reduceat(Ws * self.pos, self.starts, axis=0)
        gn = np.add.reduceat(Ws * self.neg, self.starts, axis=0)
        below = np.cumsum(gn, 0) - gn
        num = (gp * (below + 0.5 * gn)).sum(0)
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def wacc(vec):
    return lambda Wc: (vec @ Wc) / Wc.sum(0)


BASE = {}
for d, p in PRED.items():
    for a in AS:
        for c in (0, 1):
            BASE[("auc", str(d), a, c)] = WAUC(p[f"P__NL__as{a}"][:, c].astype(np.float64), YS_A == c)
CORR = {(k, arm): (HEAD[k, arm][aI].argmax(1) == YT_A).astype(float) for k in SEEDS for arm in ARMS}
UVEC = {}
for arm in ("B", "F", "F0"):
    for k in SEEDS:
        UVEC[f"acc-{arm}-A-s{k}"] = CORR[k, arm] - CORR[k, "A"]
        UVEC[f"gain-{arm}-s{k}"] = CORR[k, arm] - cK
        UVEC[f"ret-{arm}-s{k}"] = CORR[k, arm] - 0.8 * CORR[k, "A"] - 0.2 * cK
for arm in ARMS:
    for k in SEEDS:
        UVEC[f"acc-{arm}-s{k}"] = CORR[k, arm]
UVEC["acc-const"] = cK
for n, vv in UVEC.items():
    BASE[("acc", n)] = wacc(vv)

units_b, idx_b = np.unique(UNITS_A, return_inverse=True)
nU = len(units_b)
rng = np.random.default_rng(BSEED)
pu = np.full(nU, 1.0 / nU)
CNT = np.stack([rng.multinomial(nU, pu) for _ in range(NB)], axis=1)   # nU x NB, sequential RNG order
REP = {key: np.empty(NB) for key in BASE}
POINT = {}
ones = np.ones((len(aI), 1))
for key, f in BASE.items():
    POINT[key] = float(f(ones)[0])
for c0 in range(0, NB, CHUNK):
    Wc = CNT[idx_b, c0:c0 + CHUNK].astype(np.float64)
    for key, f in BASE.items():
        REP[key][c0:c0 + CHUNK] = f(Wc)
print(f"bootstrap done: {len(BASE)} base statistics x {NB} replicates, {time.time() - T0:.0f}s", flush=True)
# unweighted point AUC equals the rank-based Mann-Whitney AUC
pt_gap = max(abs(POINT[("auc", str(d), a, c)] - auc_mw(p[f"P__NL__as{a}"][:, c], YS_A == c))
             for d, p in PRED.items() for a in AS for c in (0, 1))
add("bootstrap.weighted_auc_unit_weights", "PASS" if pt_gap < 1e-12 else "FAIL",
    "weighted AUC with unit weights equals the rank-based Mann-Whitney AUC (ties 0.5) for every base statistic", pt_gap)


def stat(key):
    return POINT[key], REP[key]


def R(arm, view, surface):
    """Recovery level: mean over encoder seeds of (mean over attacker seeds of macro AUC)."""
    pts, reps = [], []
    for k in SEEDS:
        d = str(RES[uid(k, arm, view, surface)][0])
        per_as_p = [np.mean([POINT[("auc", d, a, c)] for c in (0, 1)]) for a in AS]
        per_as_r = [np.mean([REP[("auc", d, a, c)] for c in (0, 1)], axis=0) for a in AS]
        pts.append(np.mean(per_as_p))
        reps.append(np.mean(per_as_r, axis=0))
    return float(np.mean(pts)), np.mean(reps, axis=0)


def ACC(prefix):
    pts = [POINT[("acc", f"{prefix}-s{k}")] for k in SEEDS]
    reps = [REP[("acc", f"{prefix}-s{k}")] for k in SEEDS]
    return float(np.mean(pts)), np.mean(reps, axis=0)


def diff(x, y):
    return x[0] - y[0], x[1] - y[1]


def endpoint_value(e):
    i = e["id"]
    if i.startswith("U-acc-"):
        return ACC(f"acc-{e['arm']}-A")
    if i.startswith("U-gain-"):
        return ACC(f"gain-{e['arm']}")
    if i.startswith("U-retain-"):
        return ACC(f"ret-{e['arm']}")
    if i.startswith("R-out-") or i.startswith("S-out-full-"):
        sfc = e["contract"].split("/")[1]
        return diff(R(e["hi"], "out", sfc), R(e["lo"], "out", sfc))
    if i.startswith("S-feat+out-"):
        sfc = e["contract"].split("/")[1]
        return diff(R(e["hi"], "feat+out", sfc), R(e["lo"], "feat+out", sfc))
    if i.startswith("S-feat-"):
        return diff(R(e["hi"], "feat", "none"), R(e["lo"], "feat", "none"))
    if i.startswith("S-offset-out-"):
        return diff(R(e["arm"], "out", "fullbank"), R(e["arm"], "out", "iobank"))
    if i.startswith("S-bypass-"):
        return diff(R(e["arm"], "feat+clean", "full"), R(e["arm"], "feat+out", "full"))
    raise ValueError(i)


def sides(e):
    i = e["id"]
    if "hi" in e:
        v, s = {"output-only/centred": ("out", "centred"), "output-only/hard": ("out", "hard"),
                "output-only/full": ("out", "full"), "features-only": ("feat", "none")}.get(
            e["contract"], ("feat+out", e["contract"].split("/")[-1]))
        return (e["hi"], v, s), (e["lo"], v, s)
    if i.startswith("S-offset-out-"):
        return (e["arm"], "out", "fullbank"), (e["arm"], "out", "iobank")
    if i.startswith("S-bypass-"):
        return (e["arm"], "feat+clean", "full"), (e["arm"], "feat+out", "full")
    return None


TABLES = {"primary": [], "secondary": []}
flags_identical, flags_near = [], []
csv_rows = {}
for fam, fname in (("primary", "PRIMARY_USEFUL_HEAD_ENDPOINTS.csv"), ("secondary", "SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv")):
    with open(PKG / fname) as fh:
        csv_rows[fam] = list(csv.DictReader(fh))
csv_fmt_bad = []
seedcol_gap = 0.0
for fam, z in (("primary", Z_P), ("secondary", Z_S)):
    fam_lock = LOCK["families"][fam]
    run_rows = {r["id"]: r for r in INF[fam]}
    assert [e["id"] for e in fam_lock] == [r["id"] for r in INF[fam]] == [r["id"] for r in csv_rows[fam]]
    for e, crow in zip(fam_lock, csv_rows[fam]):
        pt, reps = endpoint_value(e)
        rr = reps[np.isfinite(reps)]
        se = float(np.std(rr, ddof=1))
        lo, hi = pt - z * se, pt + z * se
        dec = "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
        r = run_rows[e["id"]]
        TABLES[fam].append({"id": e["id"], "target": e["target"], "runner_point": r["point"], "replay_point": pt,
                            "runner_se": r["se"], "replay_se": se, "runner_lower": r["lower"], "replay_lower": lo,
                            "runner_upper": r["upper"], "replay_upper": hi,
                            "runner_decision": r["decision"], "replay_decision": dec,
                            "abs_diff_point": abs(pt - r["point"]), "abs_diff_se": abs(se - r["se"]),
                            "abs_diff_lower": abs(lo - r["lower"]), "runner_z": r["z"], "replay_z": z,
                            "n_finite_replicates_runner": r["n_finite_replicates"], "n_finite_replicates_replay": int(len(rr)),
                            "target_matches_lock": r["target"] == e["target"]})
        for fld, val in (("point", r["point"]), ("se", r["se"]), ("lower", r["lower"]), ("upper", r["upper"]), ("z", r["z"]),
                         ("target", e["target"])):
            if f"{val:.6f}" != crow[fld]:
                csv_fmt_bad.append(f"{e['id']}.{fld}")
        if crow["decision"] != r["decision"] or int(crow["n_finite_replicates"]) != r["n_finite_replicates"]:
            csv_fmt_bad.append(f"{e['id']}.decision/n")
        # per-seed columns of the CSV
        sd = sides(e)
        for k in SEEDS:
            if sd is None:
                pref = {"U-acc-": f"acc-{e['arm']}-A", "U-gain-": f"gain-{e['arm']}", "U-retain-": f"ret-{e['arm']}"}
                pfx = next(v for kk, v in pref.items() if e["id"].startswith(kk))
                mine = POINT[("acc", f"{pfx}-s{k}")]
            else:
                (a1, v1, s1), (a2, v2, s2) = sd
                mine = PT[uid(k, a1, v1, s1)] - PT[uid(k, a2, v2, s2)]
            seedcol_gap = max(seedcol_gap, abs(mine - float(crow[f"seed{k}"])))
        if sd is not None:
            same = all(RES[uid(k, *sd[0])][0] == RES[uid(k, *sd[1])][0] for k in SEEDS)
            if same:
                flags_identical.append(e["id"])
        if abs(lo - e["target"]) < 0.5 * se:
            flags_near.append({"id": e["id"], "family": fam, "lower": lo, "target": e["target"], "se": se,
                               "gap_over_se": (lo - e["target"]) / se, "decision": dec})

for fam in ("primary", "secondary"):
    T = TABLES[fam]
    mp = max(r["abs_diff_point"] for r in T)
    ms = max(r["abs_diff_se"] for r in T)
    ml = max(r["abs_diff_lower"] for r in T)
    dec_bad = [r["id"] for r in T if r["runner_decision"] != r["replay_decision"]]
    n_ok = all(r["n_finite_replicates_runner"] == r["n_finite_replicates_replay"] == NB for r in T)
    add(f"endpoints.{fam}.points", "PASS" if mp < 1e-9 else "FAIL", f"{len(T)} endpoint points vs inference.json", mp)
    add(f"endpoints.{fam}.se", "PASS" if ms < 1e-6 and n_ok else "FAIL",
        f"bootstrap SE (B={NB}, seed {BSEED}, ddof 1, same draw order); finite replicates all {NB}: {n_ok}", ms)
    add(f"endpoints.{fam}.lower", "PASS" if ml < 1e-6 else "FAIL", "lower bounds point - z*SE", ml)
    add(f"endpoints.{fam}.decisions", "PASS" if not dec_bad else "FAIL",
        f"replay decisions {dict(Counter(r['replay_decision'] for r in T))}; disagreements: {dec_bad}")
add("endpoints.z", "PASS" if abs(Z_P - INF["primary"][0]["z"]) < 1e-12 and abs(Z_S - INF["secondary"][0]["z"]) < 1e-12
    and round(Z_P, 6) == 3.007787 and round(Z_S, 6) == 3.171766 else "FAIL",
    f"z primary={Z_P:.9f}, secondary={Z_S:.9f}")
add("endpoints.csv_consistency", "PASS" if not csv_fmt_bad and seedcol_gap <= 5e-7 + 1e-12 else "FAIL",
    f"CSV point/se/lower/upper/z/target/decision equal inference.json at 6 decimals and seed0-2 columns equal replay "
    f"per-seed contrasts within rounding; mismatches: {csv_fmt_bad}", seedcol_gap)

# levels (descriptive)
lv_pt, lv_se = 0.0, 0.0
for name, v in INF["levels"].items():
    parts = name.split("|")
    if parts[0] == "R":
        pt, reps = R(parts[1], parts[2], parts[3])
    elif parts[1] == "const":
        pt, reps = stat(("acc", "acc-const"))
    else:
        pt, reps = ACC(f"acc-{parts[1]}")
    lv_pt = max(lv_pt, abs(pt - v["point"]))
    lv_se = max(lv_se, abs(float(np.std(reps, ddof=1)) - v["se"]))
add("levels.points_and_se", "PASS" if lv_pt < 1e-9 and lv_se < 1e-6 else "FAIL",
    f"{len(INF['levels'])} descriptive levels: max point diff {lv_pt:.3g}, max SE diff {lv_se:.3g}", max(lv_pt, lv_se))

# ================================================================================================ 7. flags
add("flags.IDENTICAL_BY_SELECTION", "WARN" if flags_identical else "PASS",
    f"endpoints whose two sides resolve to the same scored directory on every seed (difference is 0 by construction, "
    f"SE 0): {flags_identical}", endpoints=flags_identical)
add("flags.NEAR_BOUND", "WARN" if flags_near else "PASS",
    f"endpoints with |lower - target| < 0.5 SE: {[(f['id'], round(f['gap_over_se'], 3), f['decision']) for f in flags_near]}",
    endpoints=flags_near)
partial_same = []
for fam in ("primary", "secondary"):
    for e in LOCK["families"][fam]:
        sd = sides(e)
        if sd is None or e["id"] in flags_identical:
            continue
        ks = [k for k in SEEDS if RES[uid(k, *sd[0])][0] == RES[uid(k, *sd[1])][0]]
        if ks:
            partial_same.append(f"{e['id']} seeds {ks}")
add("flags.same_dir_some_seeds", "INFO", f"endpoints whose sides share a scored directory on some (not all) seeds: {partial_same}")


def preds_of_dir(d):
    d = Path(d)
    if d not in PRED:
        PRED[d] = npz(d / "preds.npz")
    return PRED[d]


def same_content(d1, d2):
    p1, p2 = preds_of_dir(d1), preds_of_dir(d2)
    return all(np.array_equal(p1[f"P__NL__as{a}"], p2[f"P__NL__as{a}"]) for a in AS)


def nl_auc_dir(d):
    p = preds_of_dir(d)
    return float(np.mean([macro_auc(p[f"P__NL__as{a}"], YS_A) for a in AS]))


# identical by content (different directories, bitwise-equal NL predictions) on every seed
content_same, content_some = [], []
for fam in ("primary", "secondary"):
    for e in LOCK["families"][fam]:
        sd = sides(e)
        if sd is None:
            continue
        ks = [k for k in SEEDS if same_content(RES[uid(k, *sd[0])][0], RES[uid(k, *sd[1])][0])]
        if len(ks) == 3:
            content_same.append(e["id"])
        elif ks:
            content_some.append(f"{e['id']} seeds {ks}")
add("flags.IDENTICAL_BY_CONTENT", "WARN" if content_same else "PASS",
    f"endpoints whose two sides have bitwise-identical saved NL predictions on every seed: {content_same}", endpoints=content_same)
add("flags.identical_content_some_seeds", "INFO",
    f"endpoints whose sides have bitwise-identical NL predictions on some seeds (per-seed contrast exactly 0): {content_some}")

# tie sensitivity: would the other tied candidate have scored different predictions?
tie_rows, tie_material = [], []
for k in SEEDS:
    for arm in ARMS:
        for (view, b), cs in BANK_CANDS.items():
            cands = [uid(k, arm, view, c) for c in cs]
            vals = [float(rec_of(c)["val_log_loss"]["NL"]) for c in cands]
            tied = [c for c, v in zip(cands, vals) if v == min(vals)]
            if len(tied) > 1:
                dirs = [resolve(c)[0] for c in tied]
                same = all(same_content(dirs[0], x) for x in dirs[1:])
                gap = max(abs(nl_auc_dir(dirs[0]) - nl_auc_dir(x)) for x in dirs[1:])
                tie_rows.append({"unit": uid(k, arm, view, b), "tied": [c.split("__", 2)[2] for c in tied],
                                 "bitwise_same_predictions": same, "max_abs_auc_gap": gap})
                if not same:
                    tie_material.append(uid(k, arm, view, b))
        j = SEL[k]["nominee_unit_source"]
        oar_rep = f"adult__s{k}__Fc{j}__rep" if arm == "F" else f"adult__s{k}__{TAG[arm]}__rep"
        for sn, rep_u, out_u in [(sn, uid(k, arm, "feat", "none"), uid(k, arm, "out", sn)) for sn in ("centred", "prob", "hard")] + \
                [("full", oar_rep, f"adult__s{k}__O_head{TAG[arm]}")]:
            u = uid(k, arm, "feat+out", sn)
            cand = {"GBT/MLP": unit_path(u), "ignore_rep": resolve(out_u)[0], "ignore_out": resolve(rep_u)[0]}
            vals = {"GBT/MLP": float(rec_of(u)["val_log_loss"]["NL"]), "ignore_rep": float(rec_of(out_u)["val_log_loss"]["NL"]),
                    "ignore_out": float(rec_of(rep_u)["val_log_loss"]["NL"])}
            tied = [n for n, v in vals.items() if v == min(vals.values())]
            if len(tied) > 1:
                dirs = [cand[n] if n != "GBT/MLP" else (CAPU / u if (CAPU / u / "preds.npz").exists() else ALIAS_SRC[u])
                        for n in tied]
                same = all(same_content(dirs[0], x) for x in dirs[1:])
                gap = max(abs(nl_auc_dir(dirs[0]) - nl_auc_dir(x)) for x in dirs[1:])
                tie_rows.append({"unit": u, "tied": tied, "bitwise_same_predictions": same, "max_abs_auc_gap": gap})
                if not same:
                    tie_material.append(u)
ENDPOINT_VS = {("out", "centred"), ("out", "hard"), ("out", "full"), ("feat+out", "full"), ("feat+out", "centred"),
               ("feat+out", "hard"), ("feat", "none"), ("out", "fullbank"), ("out", "iobank"), ("feat+clean", "full")}
tie_endpoint_material = [r["unit"] for r in tie_rows if not r["bitwise_same_predictions"] and r["max_abs_auc_gap"] > 0
                         and tuple(r["unit"].split("__")[3:5]) in ENDPOINT_VS]
for r in tie_rows:
    r["endpoint_uid"] = tuple(r["unit"].split("__")[3:5]) in ENDPOINT_VS
add("selection.tie_sensitivity", "WARN" if tie_endpoint_material else ("INFO" if tie_material else "PASS"),
    f"{len(tie_rows)} exact validation ties (banks + plus rule), all broken by candidate order as registered. The "
    f"alternative tied candidate would score different predictions in {len(tie_material)}: "
    f"{[(r['unit'], round(r['max_abs_auc_gap'], 6), 'endpoint' if r['endpoint_uid'] else 'level only') for r in tie_rows if not r['bitwise_same_predictions']]}; "
    f"ties that would change an endpoint-used recovery AUC: {tie_endpoint_material}",
    max([r["max_abs_auc_gap"] for r in tie_rows] or [0]), ties=tie_rows)

# ================================================================================================ 8. cross-study


def odx_resolve(u):
    d = ODXU / u
    r = json.loads((d / "record.json").read_text())
    if r.get("kind") == "bank":
        return odx_resolve(r["bank_selected"])
    if (d / "ALIAS.json").exists():
        return HOME / json.loads((d / "ALIAS.json").read_text())["source"].removeprefix("~/")
    return d


xs_gap, xs_bit, xs_n, xs_rows, xs_missing = 0.0, 0, 0, True, []
for k in SEEDS:
    for sn in ("centred", "dc", "prob", "hard"):
        ou = f"adult__s{k}__income_prediction__sex__RH__{sn}"
        if not (ODXU / ou).exists():
            xs_missing.append(ou)
            continue
        po = npz(odx_resolve(ou) / "preds.npz")
        pc = npz(CAPU / uid(k, "A", "out", sn) / "preds.npz")
        xs_rows &= np.array_equal(po["assess_row_id"], pc["assess_row_id"]) and np.array_equal(po["y_s"], pc["y_s"])
        a1 = np.mean([macro_auc(po[f"P__NL__as{a}"], po["y_s"]) for a in AS])
        a2 = PT[uid(k, "A", "out", sn)]
        xs_gap = max(xs_gap, abs(a1 - a2))
        xs_bit += all(np.array_equal(po[f"P__NL__as{a}"], pc[f"P__NL__as{a}"]) for a in AS)
        xs_n += 1
add("cross_study.odx_RH", "INFO",
    f"cap A output-only (centred, dc, prob, hard) vs odx RH units: {xs_n} compared, missing {xs_missing}; same assessment "
    f"rows/labels: {xs_rows}; saved NL predictions bitwise equal (all 3 attacker seeds) in {xs_bit}/{xs_n}", xs_gap)

# ================================================================================================ 9. lock
code_bad = [f for f, h in LOCK["code_files"].items() if not (WT / f).exists() or sha_file(WT / f) != h]
key_files = ("cap/run.py", "cap/family.py", "cap/lock.py", "cap/infer.py", "cap/plan.py")
add("lock.code_hashes", "PASS" if not code_bad else "FAIL",
    f"{len(LOCK['code_files'])} code files re-hashed; mismatches: {code_bad}; key cap files all match: "
    f"{not any(f in code_bad for f in key_files)}")
ct = subprocess.run(["git", "-C", str(WT), "show", "-s", "--format=%cI", LOCK_COMMIT], capture_output=True, text=True,
                    check=True).stdout.strip()
lock_t = dt.datetime.fromisoformat(ct).astimezone(dt.timezone.utc)
before, after, no_ts = [], 0, {"alias": 0, "bank": 0, "other": []}
for u in sorted(os.listdir(CAPU)):
    c = json.loads((CAPU / u / "COMPLETE.json").read_text())
    if "completed_at" not in c:
        if c.get("alias"):
            no_ts["alias"] += 1
        elif json.loads((CAPU / u / "record.json").read_text()).get("kind") == "bank":
            no_ts["bank"] += 1
        else:
            no_ts["other"].append(u)
        continue
    t = dt.datetime.strptime(c["completed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    if t > lock_t:
        after += 1
    else:
        before.append(u)
add("lock.completed_after_commit", "PASS" if not before and not no_ts["other"] and after == 84 else "FAIL",
    f"lock commit {LOCK_COMMIT[:7]} committed {lock_t.isoformat()}; {after} units with completed_at all after it; "
    f"before: {before}; without completed_at: {no_ts['alias']} alias + {no_ts['bank']} bank units, other: {no_ts['other']}")
planned = set(LOCK["units_planned"])
present = set(os.listdir(CAPU))
add("lock.units_planned", "PASS" if planned - present <= {u for u in planned if "CTL" in u} and present <= planned else "FAIL",
    f"{len(planned)} planned, {len(present)} unit dirs present; planned but absent: {sorted(planned - present)} "
    f"(controls are recorded in controls.json, not as unit dirs); present but unplanned: {sorted(present - planned)}")

# ================================================================================================ write
summ = Counter(c["status"] for c in checks)
OUT = {
    "schema": "cap-independent-verification-v1",
    "verifier": tilde(Path(__file__).resolve()),
    "verifier_sha256": sha_file(Path(__file__).resolve()),
    "verifier_imports_forbidden": list(FORBIDDEN),
    "import_guard": True,
    "forbidden_modules_loaded_at_end": [m for m in sys.modules if m.split(".")[0] in FORBIDDEN],
    "lock_commit": LOCK_COMMIT,
    "inputs": {"inference_json": tilde(PRIV / "cap_v1" / "run" / "inference.json"),
               "inference_json_sha256": sha_file(PRIV / "cap_v1" / "run" / "inference.json"),
               "labels": tilde(BENCH / "inputs" / "adult_labels.npz"),
               "labels_sha256": sha_file(BENCH / "inputs" / "adult_labels.npz")},
    "roles": {"counts": counts, "row_id_sha256_sorted": role_hashes, "assessment_groups": n_groups},
    "bootstrap": {"B": NB, "seed": BSEED, "n_units": int(nU), "counts_sha256": sha_arr(CNT.astype(np.int64)),
                  "n_base_statistics": len(BASE), "z_primary": Z_P, "z_secondary": Z_S},
    "checks": checks,
    "primary": TABLES["primary"],
    "secondary": TABLES["secondary"],
    "flags": {"IDENTICAL_BY_SELECTION": flags_identical, "NEAR_BOUND": flags_near},
    "summary": {s: int(summ.get(s, 0)) for s in ("PASS", "FAIL", "INFO", "WARN")},
    "runtime_s": round(time.time() - T0, 1),
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "scikit-learn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__,
                 "joblib": joblib.__version__},
}
assert not OUT["forbidden_modules_loaded_at_end"]
(PKG / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
print("summary", OUT["summary"], f"{OUT['runtime_s']}s")
