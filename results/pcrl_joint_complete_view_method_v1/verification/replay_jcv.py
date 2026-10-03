"""Independent replay / audit of the joint complete-view (JCV) method study.

Rebuilds every checked quantity from the raw saved arrays and manifests (private inputs npz, unit files, outer preds,
SELECTION_LOCK.json, LOCK.json) with its own code. It never imports the study code: an import guard on sys.meta_path
refuses jcv, oar, odx, cap, stored_model_eval, report and pcrl, and the end of the run asserts none was loaded.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_joint_complete_view_method_v1/verification/replay_jcv.py

Writes results/pcrl_joint_complete_view_method_v1/INDEPENDENT_VERIFICATION.json (aggregates, counts and hashes only).
"""
from __future__ import annotations

import importlib.abc
import sys

FORBIDDEN = ("jcv", "oar", "odx", "cap", "stored_model_eval", "report", "pcrl")


class _ImportGuard(importlib.abc.MetaPathFinder):
    blocked: list = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            _ImportGuard.blocked.append(fullname)
            raise ImportError(f"independent replay: forbidden import {fullname!r}")
        return None


sys.meta_path.insert(0, _ImportGuard())
assert not [m for m in sys.modules if m.split(".")[0] in FORBIDDEN], "forbidden module already loaded"

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1"

T0 = time.time()
HOME = Path.home()
WT = Path(__file__).resolve().parents[3]
PKG = WT / "results" / "pcrl_joint_complete_view_method_v1"
PRIV = HOME / "PCRL_eval_cache_private" / "jcv_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
INPUTS = PRIV / "inputs" / "adult_jcv.npz"
OUT = PKG / "INDEPENDENT_VERIFICATION.json"

SEEDS = (0, 1, 2)
ARMS = ("U", "E", "L", "J", "JP", "S12", "S21", "F", "F0")
NEURAL = ("L", "J", "JP", "S12", "S21")
BETAS = (0.1, 1.0, 10.0)
KS = (2, 6)
CLIP = 1e-12
SUPPORT_MIN = 30

CHECKS: list[dict] = []


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def check(cid, status, detail, max_abs_diff=None, **extra):
    row = {"id": cid, "status": status, "detail": detail}
    if max_abs_diff is not None:
        row["max_abs_diff"] = float(max_abs_diff)
    row.update(extra)
    CHECKS.append(row)
    log(f"{status:5s} {cid}: {detail}" + ("" if max_abs_diff is None else f" (max_abs_diff={max_abs_diff:.3g})"))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tilde(p: Path) -> str:
    try:
        return "~/" + str(Path(p).resolve().relative_to(HOME))
    except ValueError:
        return str(p)


def uname(k, arm, b=None):
    return f"nn__s{k}__{arm}" + ("" if b is None else f"__b{b:g}")


def jload(p):
    return json.loads(Path(p).read_text())


def gstr(x):
    return "None" if x is None else f"{x:.3g}"


# ====================================================================================================== data
log("loading inputs")
Z = np.load(INPUTS, allow_pickle=False)
D = {k: Z[k] for k in Z.files}
ROLE = D["role"]
IDX = {r: np.flatnonzero(ROLE == r) for r in np.unique(ROLE)}
SEX = D["sex"].astype(np.int64)
RACE = D["race"].astype(np.int64)
Y = {0: D["y_income"].astype(np.int64), 1: D["y_occupation_group"].astype(np.int64)}
X = D["X"]
ROWID = D["row_id"]
RUNIT = D["unit"]
TR, DV = IDX["defense_train"], IDX["defense_val"]
AF, AV, AS = IDX["attacker_fit"], IDX["attacker_val"], IDX["assessment"]
CONST = {j: int(np.argmax(np.bincount(Y[j][TR]))) for j in (0, 1)}
LOCK = jload(PKG / "LOCK.json")
SL = jload(PKG / "SELECTION_LOCK.json")

# ====================================================================================================== 1 roles
expect = {"defense_train": 19230, "defense_val": 4897, "attacker_fit": 6065, "attacker_val": 2235, "assessment": 5243,
          "cert": 1500}
counts = {r: int(len(ix)) for r, ix in IDX.items()}
bad = {r: (counts.get(r), n) for r, n in expect.items() if counts.get(r) != n}
check("1a_role_sizes", "PASS" if not bad else "FAIL",
      "role sizes " + ", ".join(f"{r}={counts[r]}" for r in sorted(counts)) + (f"; mismatches {bad}" if bad else ""),
      role_counts=counts)
check("1a_total_rows", "PASS" if len(ROWID) == 39205 and len(np.unique(ROWID)) == 39205 else "FAIL",
      f"{len(ROWID)} rows, {len(np.unique(ROWID))} unique row ids")

ovl = {}
dtu = set(RUNIT[TR].tolist())
for r in ("defense_val", "attacker_fit", "attacker_val", "assessment", "cert"):
    ovl[f"defense_train&{r}"] = len(dtu & set(RUNIT[IDX[r]].tolist()))
others = ("attacker_fit", "attacker_val", "assessment", "cert")
for i, a in enumerate(others):
    for b in others[i + 1:]:
        ovl[f"{a}&{b}"] = len(set(RUNIT[IDX[a]].tolist()) & set(RUNIT[IDX[b]].tolist()))
hard_ovl = {k: v for k, v in ovl.items() if k.startswith("defense_train&") and k != "defense_train&defense_val" and v}
check("1b_record_disjointness", "PASS" if not hard_ovl and all(v == 0 for v in ovl.values()) else
      ("FAIL" if hard_ovl else "WARN"),
      "shared record groups (unit) between roles: " + ", ".join(f"{k}={v}" for k, v in ovl.items()),
      overlaps=ovl)
check("1b_assessment_groups", "INFO", f"assessment: {len(AS)} rows in {len(np.unique(RUNIT[AS]))} record groups")

rh = {r: hashlib.sha256(np.sort(ROWID[IDX[r]]).astype("<i8").tobytes()).hexdigest() for r in IDX}
mm = [r for r, h in SL["role_row_hashes"].items() if rh.get(r) != h]
check("1c_role_row_hashes_vs_selection_lock", "PASS" if not mm else "FAIL",
      f"{len(SL['role_row_hashes']) - len(mm)}/{len(SL['role_row_hashes'])} role row-id hashes equal SELECTION_LOCK"
      + (f"; mismatched {mm}" if mm else ""))
ish = sha_file(INPUTS)
check("1d_private_inputs_sha256", "PASS" if ish == LOCK["admitted"]["private_inputs_sha256"] else "FAIL",
      f"inputs npz sha256 {ish[:16]}... vs LOCK admitted {LOCK['admitted']['private_inputs_sha256'][:16]}...")

# ====================================================================================================== 2 input contract
names = [str(n) for n in D["feature_names"]]
base = sorted({n.split("=")[0] for n in names})
kept = sorted(["age", "workclass", "education", "education-num", "marital-status", "relationship", "capital-gain",
               "capital-loss", "hours-per-week", "native-country"])
banned_words = ("sex", "gender", "race", "income", "occupation", "fnlwgt", "row_id", "record")
hits = [n for n in names if any(w in n.lower() for w in banned_words)]
check("2a_feature_names", "PASS" if (not hits and base == kept and X.shape[1] == 83 == len(names)) else "FAIL",
      f"{len(names)} columns; base columns = {base}; banned-name hits = {hits}")

Xd = X.astype(np.float64)
targets = {"sex": SEX, "row_id": ROWID, "income": Y[0], "occupation_group": Y[1], "race": RACE}
for c in range(6):
    targets[f"occupation_group=={c}"] = (Y[1] == c).astype(np.int64)
for c in np.unique(RACE):
    targets[f"race=={c}"] = (RACE == c).astype(np.int64)
targets["sex==0"] = (SEX == 0).astype(np.int64)
targets["income==0"] = (Y[0] == 0).astype(np.int64)


def colcorr(M, v):
    v = v.astype(np.float64)
    Mc = M - M.mean(0)
    vc = v - v.mean()
    sd = np.sqrt((Mc ** 2).mean(0)) * np.sqrt((vc ** 2).mean())
    return np.where(sd > 0, (Mc * vc[:, None]).mean(0) / np.where(sd > 0, sd, 1), 0.0)


eq_hits, corr_max = [], {}
for t, v in targets.items():
    for j in range(Xd.shape[1]):
        if np.array_equal(Xd[:, j], v.astype(np.float64)):
            eq_hits.append((names[j], t))
    cc = np.abs(colcorr(Xd, v))
    corr_max[t] = (float(cc.max()), names[int(cc.argmax())])
near1 = {t: v for t, v in corr_max.items() if v[0] > 0.999}
check("2b_no_column_equals_protected_label_or_key", "PASS" if not eq_hits and not near1 else "FAIL",
      f"exact-equality hits {eq_hits}; |corr|>0.999 hits {near1}; max |corr| with income "
      f"{corr_max['income'][0]:.3f} ({corr_max['income'][1]}), occupation_group "
      f"{max(corr_max[f'occupation_group=={c}'][0] for c in range(6)):.3f}, row_id {corr_max['row_id'][0]:.3f}")
cs = np.abs(colcorr(Xd, SEX))
top = np.argsort(-cs)[:3]
check("2c_max_abs_corr_with_sex", "INFO",
      "max |corr(X_j, SEX)| over all rows = " + ", ".join(f"{names[j]}: {cs[j]:.3f}" for j in top)
      + "; relationship=Husband/Wife is the disclosed permitted proxy",
      max_abs_corr_sex=float(cs.max()))
hw = np.isin(np.arange(X.shape[1]), [names.index("relationship=Husband"), names.index("relationship=Wife")])
share_hw = float((Xd[:, hw].sum(1) > 0).mean())
male_h = float(SEX[Xd[:, names.index("relationship=Husband")] > 0].mean())
male_w = float(SEX[Xd[:, names.index("relationship=Wife")] > 0].mean())
check("2d_relationship_shortcut", "INFO",
      f"Husband/Wife rows share {share_hw:.3f}; P(male|Husband)={male_h:.4f}, P(male|Wife)={male_w:.4f} (disclosed)")

# ====================================================================================================== 3/4 deployment replay
log("deployment-graph replay (all neural and FARE units)")
X32 = torch.from_numpy(np.ascontiguousarray(X))


def my_encode(sd, i):
    h = X32
    for j, act in ((0, True), (2, True), (4, False)):
        h = TF.linear(h, sd[f"enc.{i}.{j}.weight"], sd[f"enc.{i}.{j}.bias"])
        if act:
            h = torch.relu(h)
    return h.double().numpy()


def my_encode64(sd, i):
    h = Xd
    for j, act in ((0, True), (2, True), (4, False)):
        h = h @ sd[f"enc.{i}.{j}.weight"].double().numpy().T + sd[f"enc.{i}.{j}.bias"].double().numpy()
        if act:
            h = np.maximum(h, 0)
    return h


def my_outputs(head, R):
    z = np.asarray(head.decision_function(R), dtype=np.float64)
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], 1)
    P = head.predict_proba(R)
    return z - z.mean(1, keepdims=True), P, P.argmax(1)


def head_ok(head, nfeat, K):
    return (isinstance(head, Pipeline) and [type(s).__name__ for _, s in head.steps] ==
            ["StandardScaler", "LogisticRegression"] and int(head.n_features_in_) == nfeat and
            list(head.classes_) == list(range(K)))


def crosscov_rel(E, Zoh, Hscale):
    n = len(E)
    C = (E - E.mean(0)).T @ (Zoh - Zoh.mean(0)) / (n - 1)
    sH = Hscale.std(0, ddof=1)
    sZ = Zoh.std(0, ddof=1)
    return float(np.abs(C).max()), float((np.abs(C) / np.outer(np.where(sH > 0, sH, np.inf), sZ)).max())


ZOH = np.eye(2)[SEX]
EXPECT_KEYS = sorted([f"enc.{i}.{j}.{p}" for i in (0, 1) for j in (0, 2, 4) for p in ("weight", "bias")] +
                     [f"head.{i}.{p}" for i in (0, 1) for p in ("weight", "bias")])
dep = {"r": 0.0, "c": 0.0, "p": 0.0, "hard_mismatch": 0, "c_from_saved_r": 0.0, "p_from_saved_r": 0.0,
       "hard_from_saved_r_mismatch": 0, "h32_vs_h64": 0.0, "units": 0, "head_struct_bad": [], "keys_bad": [],
       "scaler_mean_vs_defense_train": 0.0, "scaler_n_seen_bad": [], "leace_mean_vs_defense_train_h": 0.0,
       "leace_rank": set()}
native = {"fit_rel_max": 0.0, "fit_abs_max": 0.0, "assess_rel": [], "units": 0}
NATIVE_UNIT = {}       # unit -> {i: (fit_rel, assess_heldout_corr)}
REL = {}               # unit -> saved release arrays (only those needed later are kept)
for k in SEEDS:
    names_k = [uname(k, "U"), uname(k, "E")] + [uname(k, a, b) for a in NEURAL for b in BETAS]
    for nm in names_k:
        d = UNITS / nm
        sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
        if sorted(sd.keys()) != EXPECT_KEYS or sd["enc.0.0.weight"].shape != (64, 83) or \
                sd["enc.1.4.weight"].shape != (16, 64) or sd["head.1.weight"].shape != (6, 16):
            dep["keys_bad"].append(nm)
        z = np.load(d / "release.npz")
        assert np.array_equal(z["row_id"], ROWID)
        NATIVE_UNIT[nm] = {}
        for i in (0, 1):
            h = my_encode(sd, i)
            dep["h32_vs_h64"] = max(dep["h32_vs_h64"], float(np.abs(h - my_encode64(sd, i)).max()))
            lp = d / f"leace_{i}" / "leace_map.npz"
            if lp.exists():
                L = np.load(lp)
                r = h - ((h - L["mean_x"]) @ L["proj_right"].T) @ L["proj_left"].T
                dep["leace_mean_vs_defense_train_h"] = max(dep["leace_mean_vs_defense_train_h"],
                                                          float(np.abs(L["mean_x"] - h[TR].mean(0)).max()))
                dep["leace_rank"].add(int(np.linalg.matrix_rank(L["proj_left"] @ L["proj_right"])))
                # 4: LEACE native check on fitting rows + held-out (assessment)
                fa, fr = crosscov_rel(r[TR], ZOH[TR], h[TR])
                aa, ar = crosscov_rel(r[AS], ZOH[AS], h[AS])
                native["fit_rel_max"] = max(native["fit_rel_max"], fr)
                native["fit_abs_max"] = max(native["fit_abs_max"], fa)
                native["assess_rel"].append(ar)
                native["units"] += 1
                hc = float(np.abs(colcorr(r[AS], SEX[AS])).max())
                NATIVE_UNIT[nm][i] = (fr, hc)
            else:
                assert nm.endswith("__U"), nm
                r = h
                NATIVE_UNIT[nm][i] = (None, float(np.abs(colcorr(r[AS], SEX[AS])).max()))
            head = joblib.load(d / f"head_{i}.joblib")
            if not head_ok(head, 16, KS[i]):
                dep["head_struct_bad"].append(f"{nm}/head_{i}")
            sc = head.steps[0][1]
            dep["scaler_mean_vs_defense_train"] = max(dep["scaler_mean_vs_defense_train"],
                                                      float(np.abs(sc.mean_ - z[f"r{i + 1}"][TR].mean(0)).max()))
            if int(sc.n_samples_seen_) != len(TR):
                dep["scaler_n_seen_bad"].append(f"{nm}/head_{i}")
            c, P, hd = my_outputs(head, r)
            dep["r"] = max(dep["r"], float(np.abs(r - z[f"r{i + 1}"]).max()))
            dep["c"] = max(dep["c"], float(np.abs(c - z[f"c{i + 1}"]).max()))
            dep["p"] = max(dep["p"], float(np.abs(P - z[f"p{i + 1}"]).max()))
            dep["hard_mismatch"] += int((hd != z[f"hard{i + 1}"]).sum())
            c2, P2, hd2 = my_outputs(head, z[f"r{i + 1}"])
            dep["c_from_saved_r"] = max(dep["c_from_saved_r"], float(np.abs(c2 - z[f"c{i + 1}"]).max()))
            dep["p_from_saved_r"] = max(dep["p_from_saved_r"], float(np.abs(P2 - z[f"p{i + 1}"]).max()))
            dep["hard_from_saved_r_mismatch"] += int((hd2 != z[f"hard{i + 1}"]).sum())
        dep["units"] += 1
mx = max(dep["r"], dep["c"], dep["p"])
check("3a_deployment_replay_neural",
      "PASS" if mx < 1e-9 and dep["hard_mismatch"] == 0 else "FAIL",
      f"{dep['units']} neural units x 2 recipients, all 39,205 rows, own float32 MLP forward -> .double() -> "
      f"r = h - ((h - mean_x) Pr^T) Pl^T -> head decision_function/predict_proba: max|dr|={dep['r']:.3g}, "
      f"max|dc|={dep['c']:.3g}, max|dp|={dep['p']:.3g}, hard mismatches={dep['hard_mismatch']}", max_abs_diff=mx)
check("3b_outputs_depend_only_on_r",
      "PASS" if (max(dep["c_from_saved_r"], dep["p_from_saved_r"]) == 0 and dep["hard_from_saved_r_mismatch"] == 0
                 and not dep["head_struct_bad"] and not dep["keys_bad"]) else "FAIL",
      "every deployed head is Pipeline[StandardScaler, LogisticRegression] with n_features_in_=16 (= r_i only; no "
      "label, row key or pre-erasure output enters); outputs recomputed from the saved r_i alone: "
      f"max|dc|={dep['c_from_saved_r']:.3g}, max|dp|={dep['p_from_saved_r']:.3g}, hard mismatches="
      f"{dep['hard_from_saved_r_mismatch']}; bad heads {dep['head_struct_bad']}; bad state_dicts {dep['keys_bad']}",
      max_abs_diff=max(dep["c_from_saved_r"], dep["p_from_saved_r"]))
check("3c_fit_role_provenance", "PASS" if (dep["scaler_mean_vs_defense_train"] < 1e-9 and not dep["scaler_n_seen_bad"]
                                           and dep["leace_mean_vs_defense_train_h"] < 1e-9) else "FAIL",
      f"head StandardScaler.mean_ equals mean of r_i over defense_train (max diff {dep['scaler_mean_vs_defense_train']:.3g}),"
      f" n_samples_seen_=19230 for all heads (bad: {dep['scaler_n_seen_bad']}); LEACE mean_x equals mean of h_i over "
      f"defense_train (max diff {dep['leace_mean_vs_defense_train_h']:.3g}); LEACE erasure ranks {sorted(dep['leace_rank'])}",
      max_abs_diff=max(dep["scaler_mean_vs_defense_train"], dep["leace_mean_vs_defense_train_h"]))
check("3d_float32_forward_vs_float64", "INFO",
      f"runner-style float32 encoder vs a pure float64 forward: max|dh|={dep['h32_vs_h64']:.3g} (expected float32 "
      "rounding; the release uses the float32 forward)")
ar = np.array(native["assess_rel"])
check("4a_leace_native_fit_rows", "PASS" if native["fit_rel_max"] < 1e-6 else "FAIL",
      f"{native['units']} LEACE maps (E,L,J,JP,S12,S21 x 3 seeds x betas x 2 recipients): max relative "
      f"|Cov(r, onehot SEX)|/(sd_pre(h) sd(Z)) on defense_train = {native['fit_rel_max']:.3g} "
      f"(absolute {native['fit_abs_max']:.3g})", max_abs_diff=native["fit_rel_max"])
check("4b_leace_heldout_assessment", "INFO",
      f"same statistic on held-out assessment rows: median {np.median(ar):.3g}, max {ar.max():.3g} (nonzero, small; "
      "LEACE guarantees only the fitting rows)")

log("FARE units")
fdep = {"c": 0.0, "p": 0.0, "hard": 0, "r_onehot_bad": [], "head_bad": [], "units": 0, "scaler": 0.0}
for d in sorted(UNITS.glob("fare__s*")):
    if d.name.endswith(".quarantined"):
        continue
    z = np.load(d / "release.npz")
    cells = z["cells"]
    n = int(cells.max()) + 1
    if not np.array_equal(z["r"], np.eye(n)[cells]):
        fdep["r_onehot_bad"].append(d.name)
    head = joblib.load(d / "head.joblib")
    K = KS[int(d.name.split("__p")[1][0])]
    if not head_ok(head, n, K):
        fdep["head_bad"].append(d.name)
    fdep["scaler"] = max(fdep["scaler"], float(np.abs(head.steps[0][1].mean_ - z["r"][TR].mean(0)).max()))
    c, P, hd = my_outputs(head, z["r"])
    fdep["c"] = max(fdep["c"], float(np.abs(c - z["c"]).max()))
    fdep["p"] = max(fdep["p"], float(np.abs(P - z["p"]).max()))
    fdep["hard"] += int((hd != z["hard"]).sum())
    fdep["units"] += 1
check("3e_deployment_replay_fare",
      "PASS" if max(fdep["c"], fdep["p"]) < 1e-12 and fdep["hard"] == 0 and not fdep["r_onehot_bad"]
      and not fdep["head_bad"] and fdep["scaler"] < 1e-9 else "FAIL",
      f"{fdep['units']} FARE units: r == one-hot(cells) (bad {fdep['r_onehot_bad']}); head on cells only "
      f"(bad {fdep['head_bad']}; scaler fitted on defense_train, max diff {fdep['scaler']:.3g}); recomputed "
      f"max|dc|={fdep['c']:.3g}, max|dp|={fdep['p']:.3g}, hard mismatches={fdep['hard']}",
      max_abs_diff=max(fdep["c"], fdep["p"]))

# FARE determinism across seeds (informational)
same = []
for i in (0, 1):
    for tag in ("c1", "Z1"):
        cs_ = [np.load(UNITS / f"fare__s{k}__p{i}__{tag}" / "release.npz")["cells"] for k in SEEDS]
        bij = [len(set(zip(cs_[0].tolist(), cs_[k].tolist()))) == int(cs_[0].max()) + 1 == int(cs_[k].max()) + 1
               for k in (1, 2)]
        same.append(f"p{i}/{tag}: s1 {'same' if bij[0] else 'different'}, s2 {'same' if bij[1] else 'different'}")
check("3f_fare_seed_dependence", "INFO",
      "selected FARE trees, seeds 1 and 2 vs seed 0 (same cell partition up to relabelling?): " + ", ".join(same)
      + ". Income F (c1) and both occupation trees are the same partition on all three seeds, F0 income differs only "
        "on seed 1: the F/F0 'seeds' are close to one deterministic FARE fit, so their seed spread mostly reflects "
        "cell labels, attacker refits and the per-seed U gates, not independent FARE fits")

# ====================================================================================================== 5 selection replay
log("selection replay")


def inner(name):
    return jload(UNITS / f"inner__{name}" / "record.json")


def acc_on(hard, rows, j):
    return float((hard[rows] == Y[j][rows]).mean())


const_av = {j: float((Y[j][AV] == CONST[j]).mean()) for j in (0, 1)}
iu_diff = 0.0
iu_n = 0


def my_util_nn(nm):
    z = np.load(UNITS / nm / "release.npz")
    return {j: {"acc": acc_on(z[f"hard{j + 1}"], AV, j), "const_acc": const_av[j]} for j in (0, 1)}


def my_util_fare(nm, j):
    z = np.load(UNITS / nm / "release.npz")
    return {"acc": acc_on(z["hard"], AV, j), "const_acc": const_av[j]}


def gate_vals(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
    return all(x >= 0 for x in g.values()), min(g.values())


def gates2(u, uU):
    oks, ms = zip(*(gate_vals(u[j], uU[j]) for j in (0, 1)))
    return all(oks), min(ms)


MYSEL = {}
sel_mism = []
for k in SEEDS:
    S = {"arms": {}, "fare": {}}
    uU = my_util_nn(uname(k, "U"))
    for nm in [uname(k, "U"), uname(k, "E")] + [uname(k, a, b) for a in NEURAL for b in BETAS]:
        rec = inner(nm)
        mine = my_util_nn(nm)
        for j in (0, 1):
            for q in ("acc", "const_acc"):
                iu_diff = max(iu_diff, abs(mine[j][q] - rec["utility"][str(j)][q]))
                iu_n += 1

    def cand(arm, b):
        nm = uname(k, arm, b)
        rec = inner(nm)
        ok, m = gates2(my_util_nn(nm), uU)
        return {"unit": nm, "beta": b, "gates_ok": ok, "margin": m, **{w: rec["recovery"][w] for w in ("v1", "v2", "pair")}}

    Lc = [cand("L", b) for b in BETAS]
    feas = [c for c in Lc if c["gates_ok"]]
    if feas:
        Ls = min(feas, key=lambda c: (max(c["v1"], c["v2"]), (c["v1"] + c["v2"]) / 2, c["beta"]))
        S["arms"]["L"] = {"status": "NOMINEE", **Ls}
    else:
        S["arms"]["L"] = {"status": "NO_FEASIBLE_NOMINEE", **max(Lc, key=lambda c: (c["margin"], -c["beta"]))}
    S["arms"]["L"]["n_feasible"] = len(feas)
    Lr = S["arms"]["L"]
    for arm in ("J", "JP", "S12", "S21"):
        C = [cand(arm, b) for b in BETAS]
        f2 = [c for c in C if c["gates_ok"] and Lr["status"] == "NOMINEE" and c["v1"] <= Lr["v1"] + 0.01
              and c["v2"] <= Lr["v2"] + 0.01]
        if f2:
            S["arms"][arm] = {"status": "NOMINEE", **min(f2, key=lambda c: (c["pair"], c["beta"]))}
        else:
            S["arms"][arm] = {"status": "NO_FEASIBLE_NOMINEE", **max(C, key=lambda c: (c["margin"], -c["beta"]))}
        S["arms"][arm]["n_gates_ok"] = sum(c["gates_ok"] for c in C)
    for arm in ("U", "E"):
        nm = uname(k, arm)
        rec = inner(nm)
        ok, m = gates2(my_util_nn(nm), uU)
        S["arms"][arm] = {"status": "NOMINEE" if (ok or arm == "U") else "GATES_FAILED", "unit": nm, "gates_ok": ok,
                          "margin": m, **{w: rec["recovery"][w] for w in ("v1", "v2", "pair")}}
    for i in (0, 1):
        rows = []
        for cfg in range(1, 7):
            nm = f"fare__s{k}__p{i}__c{cfg}"
            rec = inner(nm)
            mu = my_util_fare(nm, i)
            for q in ("acc", "const_acc"):
                iu_diff = max(iu_diff, abs(mu[q] - rec["utility"][q]))
                iu_n += 1
            ok, m = gate_vals(mu, uU[i])
            rows.append({"config": cfg, "unit": nm, "gate_ok": ok, "margin": m, "R_local": rec["recovery_local"]})
        adm = [x for x in rows if x["gate_ok"]]
        if adm:
            S["fare"][i] = {"status": "NOMINEE", **min(adm, key=lambda x: (x["R_local"], x["config"]))}
        else:
            S["fare"][i] = {"status": "NO_FEASIBLE_NOMINEE", **max(rows, key=lambda x: (x["margin"], -x["config"]))}
        S["fare"][i]["table"] = rows
    for arm, tag in (("F", "c"), ("F0", "Z")):
        n1, n2 = f"fare__s{k}__p0__{tag}{S['fare'][0]['config']}", f"fare__s{k}__p1__{tag}{S['fare'][1]['config']}"
        rec = jload(UNITS / f"inner__pair__s{k}__{arm}" / "record.json")
        if rec["of"] != [n1, n2]:
            sel_mism.append(f"s{k} {arm} inner pair record of {rec['of']} != {[n1, n2]}")
        mu = {0: my_util_fare(n1, 0), 1: my_util_fare(n2, 1)}
        for j in (0, 1):
            for q in ("acc", "const_acc"):
                iu_diff = max(iu_diff, abs(mu[j][q] - rec["utility"][str(j)][q]))
                iu_n += 1
        ok, m = gates2(mu, uU)
        both = all(S["fare"][i]["status"] == "NOMINEE" for i in (0, 1))
        if arm == "F":
            st = "NOMINEE" if ok and both else "NO_FEASIBLE_NOMINEE"
        else:   # protocol: F0 = gamma-0 twin at F's selected config; undefined when F has no selected config
            st = "NOMINEE" if ok and both else ("NO_SELECTED_F_CONFIG" if not both else "GATES_FAILED")
        S["arms"][arm] = {"status": st, "units": [n1, n2], "gates_ok": ok, "margin": m,
                          "configs": [S["fare"][0]["config"], S["fare"][1]["config"]],
                          **{w: rec["recovery"][w] for w in ("v1", "v2", "pair")}}
    cands = []
    if Lr["status"] == "NOMINEE":
        for o, arm in enumerate(("E", "L", "JP", "S12", "S21", "F", "F0")):
            a = S["arms"][arm]
            if a["status"] == "NOMINEE" and a["gates_ok"] and a["v1"] <= Lr["v1"] + 0.01 and a["v2"] <= Lr["v2"] + 0.01:
                cands.append((a["pair"], o, arm))
    S["cstar"] = min(cands)[2] if cands else None
    MYSEL[k] = S

check("5a_inner_utility_from_release", "PASS" if iu_diff < 1e-12 else "FAIL",
      f"{iu_n} inner utility numbers (attacker_val accuracy of the deployed hard decisions and the defense_train-"
      f"majority constant) recomputed from release.npz and compared with the inner records", max_abs_diff=iu_diff)

st_cmp, val_diff = [], 0.0
f0_notes = []
for k in SEEDS:
    sl = SL["seeds"][str(k)]
    S = MYSEL[k]
    for arm in ARMS:
        a, b = S["arms"][arm], sl["arms"][arm]
        mine_st = a["status"]
        if arm == "F0" and mine_st == "NO_SELECTED_F_CONFIG":
            f0_notes.append(f"s{k}: lock {b['status']} with gates_ok={b['gates_ok']} (margin {b['worst_gate_margin']:.4f})"
                            f"; replay gates_ok={a['gates_ok']} (margin {a['margin']:.4f})")
            mine_st = "GATES_FAILED"  # compared separately below; runner label documented in the lock annotation
        if mine_st != b["status"]:
            st_cmp.append(f"s{k} {arm}: replay {a['status']} vs lock {b['status']}")
        if a.get("unit") != b.get("unit") or a.get("units") != b.get("units"):
            st_cmp.append(f"s{k} {arm}: unit replay {a.get('unit', a.get('units'))} vs lock {b.get('unit', b.get('units'))}")
        if a.get("beta") != b.get("beta") and arm in NEURAL:
            st_cmp.append(f"s{k} {arm}: beta replay {a.get('beta')} vs lock {b.get('beta')}")
        if a["gates_ok"] != b["gates_ok"]:
            st_cmp.append(f"s{k} {arm}: gates_ok replay {a['gates_ok']} vs lock {b['gates_ok']}")
        val_diff = max(val_diff, abs(a["margin"] - b["worst_gate_margin"]),
                       *(abs(a[w] - b[f"R_{w}"]) for w in ("v1", "v2", "pair")))
        if arm in ("F", "F0") and a["configs"] != b["configs"]:
            st_cmp.append(f"s{k} {arm}: configs replay {a['configs']} vs lock {b['configs']}")
    for i in (0, 1):
        a, b = S["fare"][i], sl["fare_selection"][str(i)]
        if (a["status"], a["config"], a["unit"]) != (b["status"], b["config"], b["unit"]):
            st_cmp.append(f"s{k} FARE p{i}: replay {(a['status'], a['config'])} vs lock {(b['status'], b['config'])}")
        val_diff = max(val_diff, abs(a["margin"] - b["margin"]), abs(a["R_local"] - b["R_local"]))
        for ra, rb in zip(a["table"], b["table"]):
            if ra["gate_ok"] != rb["gate_ok"] or ra["config"] != rb["config"]:
                st_cmp.append(f"s{k} FARE p{i} c{ra['config']}: gate replay {ra['gate_ok']} vs lock {rb['gate_ok']}")
            val_diff = max(val_diff, abs(ra["margin"] - rb["margin"]), abs(ra["R_local"] - rb["R_local"]))
    if S["cstar"] != sl["comparator"]["arm"]:
        st_cmp.append(f"s{k} C*: replay {S['cstar']} vs lock {sl['comparator']['arm']}")

summ = "; ".join(
    f"s{k}: L {MYSEL[k]['arms']['L']['status']} (descr. {MYSEL[k]['arms']['L']['unit']}, {MYSEL[k]['arms']['L']['n_feasible']}/3 "
    f"feasible), J {MYSEL[k]['arms']['J']['unit'].split('__')[-1]}, JP {MYSEL[k]['arms']['JP']['unit'].split('__')[-1]}, "
    f"S12 {MYSEL[k]['arms']['S12']['unit'].split('__')[-1]}, S21 {MYSEL[k]['arms']['S21']['unit'].split('__')[-1]}, "
    f"E {MYSEL[k]['arms']['E']['status']}, F p0 c{MYSEL[k]['fare'][0]['config']} {MYSEL[k]['fare'][0]['status']}, "
    f"F p1 c{MYSEL[k]['fare'][1]['config']} {MYSEL[k]['fare'][1]['status']}, C*={MYSEL[k]['cstar']}" for k in SEEDS)
check("5b_selection_replay_vs_lock", "PASS" if not st_cmp and val_diff < 1e-12 else "FAIL",
      "statuses, descriptive closest-utility units/betas (largest worst-gate margin, ties -> smaller beta / lower id), "
      "FARE per-purpose selections and C* recomputed from inner records + own gates: "
      + ("all match SELECTION_LOCK. " if not st_cmp else f"MISMATCHES {st_cmp}. ") + summ,
      max_abs_diff=val_diff)
gok = {k: {a: MYSEL[k]["arms"][a]["n_gates_ok"] for a in ("J", "JP", "S12", "S21")} for k in SEEDS}
check("5c_gate_feasibility_counts", "INFO",
      f"configurations passing G1-G3 on attacker_val (of 3 betas): L {[MYSEL[k]['arms']['L']['n_feasible'] for k in SEEDS]}, "
      f"others {gok}; E gates_ok {[MYSEL[k]['arms']['E']['gates_ok'] for k in SEEDS]}; FARE occupation configs passing "
      f"{[sum(r['gate_ok'] for r in MYSEL[k]['fare'][1]['table']) for k in SEEDS]}/6, income "
      f"{[sum(r['gate_ok'] for r in MYSEL[k]['fare'][0]['table']) for k in SEEDS]}/6")
check("5d_F0_status_label", "WARN" if f0_notes else "PASS",
      "F0's own pair gates pass on every seed but the lock labels it GATES_FAILED: select.finish_fare sets "
      "status = NOMINEE only if gates pass AND both F purposes have NOMINEE configs, else 'GATES_FAILED' for F0. Root "
      "cause: F has no feasible occupation_group configuration, so 'F's selected configuration' is undefined and F0 is "
      "anchored to F's descriptive config 1; the protocol has no status for this case and the code reuses "
      "GATES_FAILED. The lock's annotation discloses it. No decision effect (L has no nominee, so C* is None "
      "regardless). Per seed: " + "; ".join(f0_notes))

# runner selection.json vs SELECTION_LOCK and the public inner table
rs = jload(RUN / "selection.json")
d_rs = []
for k in SEEDS:
    for arm in ARMS:
        a, b = rs[str(k)]["arms"][arm], SL["seeds"][str(k)]["arms"][arm]
        for f in ("status", "unit", "units", "beta", "gates_ok", "worst_gate_margin", "R_v1", "R_v2", "R_pair"):
            if a.get(f) != b.get(f):
                d_rs.append(f"s{k}/{arm}/{f}")
    if rs[str(k)]["comparator"]["arm"] != SL["seeds"][str(k)]["comparator"]["arm"]:
        d_rs.append(f"s{k}/C*")
ist = list(csv.DictReader(open(PKG / "INNER_SELECTION_TABLE.csv")))
ist_bad = 0
ist_diff = 0.0
for row in ist:
    k = int(row["seed"])
    arm = row["arm"]
    if arm.startswith("F("):
        i = int(arm[3])
        t = next(x for x in MYSEL[k]["fare"][i]["table"] if x["unit"] == row["unit"])
        ist_bad += (row["gates_ok"] == "True") != t["gate_ok"]
        ist_diff = max(ist_diff, abs(float(row["worst_gate_margin"]) - t["margin"]))
        continue
    a = MYSEL[k]["arms"][arm]
    if row["unit"]:
        if arm in NEURAL:
            nm = row["unit"]
            rec = inner(nm)
            ok, m = gates2(my_util_nn(nm), my_util_nn(uname(k, "U")))
            vals = {"v1": rec["recovery"]["v1"], "v2": rec["recovery"]["v2"], "pair": rec["recovery"]["pair"]}
        else:
            ok, m, vals = a["gates_ok"], a["margin"], a
    else:
        ok, m, vals = a["gates_ok"], a["margin"], a
    ist_bad += (row["gates_ok"] == "True") != ok
    ist_bad += row["arm_status"] != SL["seeds"][str(k)]["arms"][arm]["status"]
    exp_sel = row["unit"] == SL["seeds"][str(k)]["arms"][arm].get("unit") and \
        SL["seeds"][str(k)]["arms"][arm]["status"] == "NOMINEE"
    ist_bad += (row["selected"] == "True") != exp_sel
    ist_diff = max(ist_diff, abs(float(row["worst_gate_margin"]) - m),
                   *(abs(float(row[f"R_{w}"]) - vals[w]) for w in ("v1", "v2", "pair")))
check("5e_selection_json_and_inner_table", "PASS" if not d_rs and ist_bad == 0 and ist_diff < 1e-12 else "FAIL",
      f"runner selection.json vs SELECTION_LOCK: {len(d_rs)} field differences {d_rs[:5]}; INNER_SELECTION_TABLE.csv "
      f"{len(ist)} rows vs replay: {ist_bad} gate/status/selected disagreements (only the U rows may be 'selected': "
      f"no other arm has a nominee)", max_abs_diff=ist_diff)

# ---- inner-recovery spot replay (refit the registered inner slate for two descriptive units, seed 0)
log("inner-recovery spot replay")


def est(name, seed):
    if name.startswith("LR_C"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(name[4:]), max_iter=3000))
    if name.startswith("MLP_"):
        h = tuple(int(x) for x in name[4:].split("x"))
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                             early_stopping=True, validation_fraction=0.1,
                                                             n_iter_no_change=15, random_state=seed))
    if name.startswith("HGB_"):
        _, lr, lv = name.split("_")
        return HistGradientBoostingClassifier(learning_rate=float(lr), max_leaf_nodes=int(lv), max_iter=200,
                                              early_stopping=False, random_state=seed)
    if name == "DA_canonical_MLP":
        return MyDA(seed)
    if name.startswith("CC_alpha"):
        return MyCC(float(name[8:]))
    raise ValueError(name)


class MyDA:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, Xf, yf):
        self.pca = PCA(whiten=True, random_state=self.seed).fit(Xf)
        ev = self.pca.explained_variance_
        self.keep = ev > 1e-9 * ev.max()
        self.m = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m.fit(self.pca.transform(Xf)[:, self.keep], yf)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, Xs):
        return self.m.predict_proba(self.pca.transform(Xs)[:, self.keep])


class MyCC:
    """Exact cell-conditional Laplace attacker for finite releases (own implementation)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, Xf, yf):
        self.K = max(2, int(yf.max()) + 1)
        keys = [r.tobytes() for r in np.ascontiguousarray(np.round(Xf, 9))]
        self.prior = np.bincount(yf, minlength=self.K) / len(yf)
        self.tab = {}
        for key, s in zip(keys, yf):
            self.tab.setdefault(key, np.zeros(self.K))[s] += 1
        self.classes_ = np.arange(self.K)
        return self

    def predict_proba(self, Xs):
        out = np.empty((len(Xs), self.K))
        for i, r in enumerate(np.ascontiguousarray(np.round(Xs, 9))):
            c = self.tab.get(r.tobytes())
            out[i] = self.prior if c is None else (c + self.alpha * self.prior) / (c.sum() + self.alpha)
        return out


def proba_k(m, Xs, K):
    P = m.predict_proba(Xs)
    out = np.zeros((len(Xs), K))
    for j, c in enumerate(m.classes_):
        out[:, int(c)] = P[:, j]
    return out


def ll(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1))))


def views_nn(nm):
    z = np.load(UNITS / nm / "release.npz")
    v1 = np.hstack([z["r1"], z["c1"]])
    v2 = np.hstack([z["r2"], z["c2"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "p1": z["p1"], "p2": z["p2"], "ppair": np.hstack([z["p1"], z["p2"]]),
            "h1": np.eye(2)[z["hard1"]], "h2": np.eye(6)[z["hard2"]],
            "hpair": np.hstack([np.eye(2)[z["hard1"]], np.eye(6)[z["hard2"]]])}


def views_fare(n1, n2):
    a, b = np.load(UNITS / n1 / "release.npz"), np.load(UNITS / n2 / "release.npz")
    v1 = np.hstack([a["r"], a["c"]])
    v2 = np.hstack([b["r"], b["c"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "p1": a["p"], "p2": b["p"], "ppair": np.hstack([a["p"], b["p"]]),
            "h1": np.eye(2)[a["hard"]], "h2": np.eye(6)[b["hard"]],
            "hpair": np.hstack([np.eye(2)[a["hard"]], np.eye(6)[b["hard"]]])}


INNER_SLATE = ("LR_C1", "MLP_64x64", "HGB_0.1_31")


def inner_fit(Xv):
    best = None
    for nm in INNER_SLATE:
        m = est(nm, 0).fit(Xv[AF], SEX[AF])
        P = proba_k(m, Xv[AV], 2)
        l_ = ll(SEX[AV], P)
        if best is None or l_ < best[0] - 1e-12:
            best = (l_, nm, P)
    return best


ir_diff = 0.0
ir_rows = []
for nm in (SL["seeds"]["0"]["arms"]["L"]["unit"], SL["seeds"]["0"]["arms"]["J"]["unit"]):
    V = views_nn(nm)
    rec = inner(nm)["recovery"]
    b1, b2, bp = inner_fit(V["v1"]), inner_fit(V["v2"]), inner_fit(V["pair"])
    cands = [("pair:" + bp[1], bp[0], bp[2]), ("ignore_other:v1:" + b1[1], b1[0], b1[2]),
             ("ignore_other:v2:" + b2[1], b2[0], b2[2])]
    best = min(cands, key=lambda c: c[1])
    mine = {"v1": roc_auc_score(SEX[AV], b1[2][:, 1]), "v2": roc_auc_score(SEX[AV], b2[2][:, 1]),
            "pair": roc_auc_score(SEX[AV], best[2][:, 1])}
    for w in mine:
        ir_diff = max(ir_diff, abs(mine[w] - rec[w]))
    ir_rows.append(f"{nm}: pair selected {best[0]} (record {rec['pair_selected']})")
    if best[0] != rec["pair_selected"]:
        ir_rows[-1] += " MISMATCH"
check("5f_inner_recovery_spot_refit", "PASS" if ir_diff < 1e-9 and "MISMATCH" not in " ".join(ir_rows) else "FAIL",
      "inner slate (LR_C1, MLP_64x64, HGB_0.1_31; attacker_fit -> attacker_val log loss; pair bank + ignore-other-view) "
      "refitted for the seed-0 L and J descriptive units; v1/v2/pair attacker_val AUC vs inner records. "
      + "; ".join(ir_rows), max_abs_diff=ir_diff)

# ====================================================================================================== 6 outer recomputation
log("loading outer preds")
PR = {}
pr_bad = []
for k in SEEDS:
    for arm in ARMS:
        z = np.load(UNITS / f"outer__s{k}__{arm}" / "preds.npz")
        PR[(k, arm)] = {q: z[q] for q in z.files}
p0 = PR[(0, "U")]
A_ROWS = p0["assess_row_id"]
for key, p in PR.items():
    if not (np.array_equal(p["assess_row_id"], ROWID[AS]) and np.array_equal(p["assess_unit"], RUNIT[AS])
            and np.array_equal(p["sex"], SEX[AS]) and np.array_equal(p["y_income"], Y[0][AS])
            and np.array_equal(p["y_occ"], Y[1][AS])):
        pr_bad.append(f"{key}: assessment rows/labels")
    for q in ("P_v1", "P_v2", "P_pair", "P_p1", "P_p2", "P_ppair", "P_h1", "P_h2", "P_hpair"):
        P = p[q]
        if P.shape != (3, len(AS), 2) or not np.isfinite(P).all() or np.abs(P.sum(2) - 1).max() > 1e-9:
            pr_bad.append(f"{key}:{q}")
check("1e_outer_assessment_rows", "PASS" if not pr_bad else "FAIL",
      f"27 outer preds: assess_row_id, assess_unit, SEX and task labels equal the assessment role rows in order; all "
      f"P_<view> arrays (3 x {len(AS)} x 2) finite and normalised; problems {pr_bad}")

# race subset
Kr = int(RACE.max()) + 1
rsup = [c for c in range(Kr) if all((RACE[IDX[r]] == c).sum() >= SUPPORT_MIN for r in ("attacker_fit", "attacker_val",
                                                                                        "assessment"))]
keep = np.isin(RACE[AS], rsup)
race_bad = []
for key, p in PR.items():
    if not (np.array_equal(p["race_row_id"], ROWID[AS][keep]) and np.array_equal(p["race"], RACE[AS][keep])):
        race_bad.append(str(key))
RIDX = np.flatnonzero(keep)
check("1f_race_subset", "PASS" if not race_bad else "FAIL",
      f"race classes with >=30 rows in attacker_fit, attacker_val and assessment: {rsup}; subset {keep.sum()} of "
      f"{len(AS)} assessment rows; preds race rows equal this subset in every outer unit (bad {race_bad})")

# preds outputs come from the locked units
src_diff, src_bad = 0.0, []
for (k, arm), p in PR.items():
    a = SL["seeds"][str(k)]["arms"][arm]
    if "units" in a:
        za, zb = np.load(UNITS / a["units"][0] / "release.npz"), np.load(UNITS / a["units"][1] / "release.npz")
        ref = {"p1": za["p"], "p2": zb["p"], "hard1": za["hard"], "hard2": zb["hard"]}
    else:
        z = np.load(UNITS / a["unit"] / "release.npz")
        ref = {q: z[q] for q in ("p1", "p2", "hard1", "hard2")}
    for q in ("p1", "p2"):
        src_diff = max(src_diff, float(np.abs(p[q] - ref[q][AS]).max()))
    for q in ("hard1", "hard2"):
        if not np.array_equal(p[q], ref[q][AS]):
            src_bad.append(f"s{k}/{arm}/{q}")
check("10e_outer_outputs_from_locked_units", "PASS" if src_diff == 0 and not src_bad else "FAIL",
      "deployed p1/p2/hard1/hard2 in every outer preds file equal the SELECTION_LOCK unit's release.npz at the assessment "
      f"rows (bad {src_bad})", max_abs_diff=src_diff)

# ---- weighted statistics
n = len(AS)
uu, uidx = np.unique(p0["assess_unit"], return_inverse=True)
NU = len(uu)
B = int(LOCK["families"]["B"])
BSEED = int(LOCK["families"]["boot_seed"])
assert (B, BSEED) == (1999, 20261013)
rng = np.random.default_rng(BSEED)
pu = np.full(NU, 1.0 / NU)
COUNTS = np.empty((NU, B), dtype=np.float64)
for b in range(B):
    COUNTS[:, b] = rng.multinomial(NU, pu)


class WAUC:
    """Weighted AUC = weighted fraction of correctly ordered (positive, negative) pairs, ties counted 0.5."""

    def __init__(self, score, pos, rows=None):
        o = np.argsort(score, kind="mergesort")
        s = score[o]
        self.gidx = o if rows is None else np.asarray(rows)[o]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        self.pos = pos[o].astype(np.float64)

    def __call__(self, WT):
        W = WT[self.gidx]
        Wp = W * self.pos[:, None]
        Wn = W - Wp
        gp = np.add.reduceat(Wp, self.starts, axis=0)
        gn = np.add.reduceat(Wn, self.starts, axis=0)
        below = np.cumsum(gn, axis=0) - gn
        num = (gp * (below + 0.5 * gn)).sum(0)
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


class WMean:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float64)

    def __call__(self, WT):
        return (self.v @ WT) / WT.sum(0)


BASE = {}          # sid -> callable
CONTENT = {}       # content hash -> sid


def base(sid, fn, content_key):
    if content_key in CONTENT:
        return CONTENT[content_key]
    BASE[sid] = fn
    CONTENT[content_key] = sid
    return sid


VIEW = {("prim", "v1"): "P_v1", ("prim", "v2"): "P_v2", ("prim", "pair"): "P_pair",
        ("prob", "v1"): "P_p1", ("prob", "v2"): "P_p2", ("prob", "pair"): "P_ppair",
        ("hard", "v1"): "P_h1", ("hard", "v2"): "P_h2", ("hard", "pair"): "P_hpair"}
SEXA = p0["sex"]
RACEA = p0["race"]
race_classes = sorted(set(RACEA.tolist()))
REC = {}    # (k, arm, fmt, view) -> list of 3 sids
RREC = {}   # (k, arm, view) -> list of 3 lists of class sids
ACC = {}    # (k, arm, j) -> sid
CONSTS = {}
for (k, arm), p in PR.items():
    for (fmt, w), q in VIEW.items():
        sids = []
        for s in range(3):
            sc = np.ascontiguousarray(p[q][s][:, 1])
            ck = "sex|" + hashlib.sha256(sc.tobytes()).hexdigest()
            sids.append(base(f"auc|{k}|{arm}|{q}|{s}", WAUC(sc, SEXA == 1), ck))
        REC[(k, arm, fmt, w)] = sids
    for w in ("v1", "v2", "pair"):
        P3 = p[f"Prace_{w}"]
        per = []
        for s in range(3):
            cl = []
            for c in race_classes:
                sc = np.ascontiguousarray(P3[s][:, c])
                ck = f"race{c}|" + hashlib.sha256(sc.tobytes()).hexdigest()
                cl.append(base(f"rauc|{k}|{arm}|{w}|{s}|{c}", WAUC(sc, RACEA == c, rows=RIDX), ck))
            per.append(cl)
        RREC[(k, arm, w)] = per
    for j in (0, 1):
        y = p["y_income"] if j == 0 else p["y_occ"]
        corr = (p[f"hard{j + 1}"] == y).astype(np.float64)
        ACC[(k, arm, j)] = base(f"acc|{k}|{arm}|{j}", WMean(corr), f"acc|" + hashlib.sha256(corr.tobytes()).hexdigest())
for j in (0, 1):
    y = p0["y_income"] if j == 0 else p0["y_occ"]
    v = (y == CONST[j]).astype(np.float64)
    CONSTS[j] = base(f"const|{j}", WMean(v), f"const|{j}")
log(f"{len(BASE)} distinct base statistics; bootstrap B={B} over {NU} record groups")

# point estimates
ones = np.ones((n, 1))
PT = {sid: float(fn(ones)[0]) for sid, fn in BASE.items()}
# sanity: unweighted AUC vs sklearn
sk_diff = 0.0
for (k, arm, fmt, w), sids in list(REC.items())[:40]:
    for s, sid in enumerate(sids):
        sk_diff = max(sk_diff, abs(PT[sid] - roc_auc_score(SEXA == 1, PR[(k, arm)][VIEW[(fmt, w)]][s][:, 1])))
for (k, arm, w), per in list(RREC.items())[:10]:
    for s, cl in enumerate(per):
        for c, sid in zip(race_classes, cl):
            sk_diff = max(sk_diff, abs(PT[sid] - roc_auc_score(RACEA == c, PR[(k, arm)][f"Prace_{w}"][s][:, c])))
check("6a_weighted_auc_selftest", "PASS" if sk_diff < 1e-12 else "FAIL",
      "own weighted pair-ordering AUC at unit weights vs sklearn roc_auc_score (ties 0.5) on a sample of base stats",
      max_abs_diff=sk_diff)

REP = {sid: np.empty(B) for sid in BASE}
CH = 200
for b0 in range(0, B, CH):
    Wb = COUNTS[uidx, b0:b0 + CH]
    for sid, fn in BASE.items():
        REP[sid][b0:b0 + CH] = fn(Wb)
log("bootstrap done")


def mean_list(vals):
    return np.mean(np.stack(vals), axis=0)


def get(sid, boot):
    return REP[sid] if boot else np.array([PT[sid]])


def R_(k, arm, fmt, w, boot):
    return mean_list([get(s, boot) for s in REC[(k, arm, fmt, w)]])


def RR_(k, arm, w, boot):
    return mean_list([mean_list([get(c, boot) for c in cl]) for cl in RREC[(k, arm, w)]])


def A_(k, arm, j, boot):
    return get(ACC[(k, arm, j)], boot)


def C_(j, boot):
    return get(CONSTS[j], boot)


def cstar(k):
    return SL["seeds"][str(k)]["comparator"]["arm"]


def endpoint(e, boot):
    per = []
    for k in SEEDS:
        kind = e["kind"]
        if kind in ("coalition", "local"):
            ref = cstar(k) if e["ref"] == "C*" else e["ref"]
            if ref is None:
                return None
            if kind == "coalition":
                per.append(R_(k, ref, "prim", "pair", boot) - R_(k, "J", "prim", "pair", boot))
            else:
                per.append(R_(k, "J", "prim", e["view"], boot) - R_(k, ref, "prim", e["view"], boot))
        elif kind == "acc":
            per.append(A_(k, "J", e["task"], boot) - A_(k, "U", e["task"], boot))
        elif kind == "retain":
            per.append(A_(k, "J", e["task"], boot) - 0.8 * A_(k, "U", e["task"], boot) - 0.2 * C_(e["task"], boot))
        elif kind == "useful":
            per.append(A_(k, "J", e["task"], boot) - C_(e["task"], boot))
        elif kind == "out":
            per.append(R_(k, e["a"], e["fmt"], e["view"], boot) - R_(k, e["b"], e["fmt"], e["view"], boot))
        elif kind == "race":
            per.append(RR_(k, e["a"], e["view"], boot) - RR_(k, e["b"], e["view"], boot))
        elif kind == "rec":
            per.append(R_(k, e["a"], "prim", e["view"], boot) - R_(k, e["b"], "prim", e["view"], boot))
        elif kind == "accdiff":
            per.append(A_(k, e["a"], e["task"], boot) - A_(k, e["b"], e["task"], boot))
        elif kind == "synergy":
            mx_ = np.maximum(R_(k, e["arm"], "prim", "v1", boot), R_(k, e["arm"], "prim", "v2", boot))
            per.append(R_(k, e["arm"], "prim", "pair", boot) - mx_)
        else:
            raise ValueError(kind)
    return mean_list(per)


ZP = float(norm.ppf(1 - 0.05 / 36))
ZS = float(norm.ppf(1 - 0.05 / 60))
check("6b_critical_values", "PASS" if round(ZP, 6) == LOCK["families"]["z_primary"] and
      round(ZS, 6) == LOCK["families"]["z_secondary"] else "FAIL",
      f"z_primary = Phi^-1(1-0.05/36) = {ZP:.6f}, z_secondary = Phi^-1(1-0.05/60) = {ZS:.6f} (LOCK "
      f"{LOCK['families']['z_primary']}, {LOCK['families']['z_secondary']})")

INF = jload(RUN / "inference.json")


def csv_rows(fn):
    return {r["id"]: r for r in csv.DictReader(open(PKG / fn))}


def decide(e, lo, hi):
    if e["side"] == "lower>":
        return "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
    return "PASS" if hi < e["target"] else "NOT_ESTABLISHED"


MINE = {}
COMP = {"primary": [], "secondary": []}
agg = {"point": 0.0, "se": 0.0, "csv_point": 0.0, "csv_se": 0.0, "dec_mismatch": [], "bound": 0.0}
for fam, key, z_ in ((LOCK["families"]["primary"], "primary", ZP), (LOCK["families"]["secondary"], "secondary", ZS)):
    runner = {e["id"]: e for e in INF[key]}
    rcsv = csv_rows("PRIMARY_ENDPOINTS.csv" if key == "primary" else "SECONDARY_ENDPOINTS.csv")
    assert len(fam) == (18 if key == "primary" else 30) == len(runner) == len(rcsv)
    for e in fam:
        pt = endpoint(e, False)
        if pt is None:
            m = {"point": None, "se": None, "lower": None, "upper": None, "decision": "NOT_ESTABLISHED"}
        else:
            r = endpoint(e, True)
            r = r[np.isfinite(r)]
            se = float(np.std(r, ddof=1))
            pt = float(pt[0])
            lo, hi = pt - z_ * se, pt + z_ * se
            m = {"point": pt, "se": se, "lower": lo, "upper": hi, "decision": decide(e, lo, hi), "n_finite": int(len(r))}
        MINE[e["id"]] = m
        ru = runner[e["id"]]
        rc = rcsv[e["id"]]
        if m["point"] is not None and ru.get("point") is not None:
            agg["point"] = max(agg["point"], abs(m["point"] - ru["point"]))
            agg["se"] = max(agg["se"], abs(m["se"] - ru["se"]))
            agg["bound"] = max(agg["bound"], abs(m["lower"] - ru["lower"]), abs(m["upper"] - ru["upper"]))
            agg["csv_point"] = max(agg["csv_point"], abs(m["point"] - float(rc["point"])))
            agg["csv_se"] = max(agg["csv_se"], abs(m["se"] - float(rc["se"])))
        elif (m["point"] is None) != (ru.get("point") is None) or (m["point"] is None) != (rc["point"] == ""):
            agg["dec_mismatch"].append(f"{e['id']}: point availability")
        if m["decision"] != ru["decision"] or m["decision"] != rc["decision"]:
            agg["dec_mismatch"].append(f"{e['id']}: replay {m['decision']} vs runner {ru['decision']}/{rc['decision']}")
        COMP[key].append({"id": e["id"], "runner_point": ru.get("point"), "replay_point": m["point"],
                          "runner_se": ru.get("se"), "replay_se": m["se"], "runner_decision": ru["decision"],
                          "replay_decision": m["decision"], "target": e["target"], "side": e["side"],
                          "replay_lower": m["lower"], "replay_upper": m["upper"]})
check("6c_endpoint_points", "PASS" if agg["point"] < 1e-9 else "FAIL",
      f"48 endpoints (18 primary incl. 3 C*-undefined, 30 secondary) recomputed from outer preds only; max |point diff| "
      f"vs inference.json {agg['point']:.3g}, vs CSV (6 dp) {agg['csv_point']:.3g}", max_abs_diff=agg["point"])
check("6d_endpoint_ses", "PASS" if agg["se"] < 1e-9 else "FAIL",
      f"own paired multinomial bootstrap (B={B}, seed {BSEED}, {NU} record groups, ddof 1): max |SE diff| vs "
      f"inference.json {agg['se']:.3g}, vs CSV {agg['csv_se']:.3g}; max |bound diff| {agg['bound']:.3g}",
      max_abs_diff=agg["se"])
check("6e_endpoint_decisions", "PASS" if not agg["dec_mismatch"] else "FAIL",
      f"decisions ('lower>' iff lower > target, 'upper<' iff upper < target) vs runner: "
      f"{'all 48 agree' if not agg['dec_mismatch'] else agg['dec_mismatch']}; primary PASS ids "
      f"{[i for i in LOCK['families']['primary_ids'] if MINE[i]['decision'] == 'PASS']}; secondary PASS ids "
      f"{[i for i in LOCK['families']['secondary_ids'] if MINE[i]['decision'] == 'PASS']}")
near = []
for e_ in LOCK["families"]["primary"] + LOCK["families"]["secondary"]:
    m_ = MINE[e_["id"]]
    if m_["point"] is None:
        continue
    bound = m_["lower"] if e_["side"] == "lower>" else m_["upper"]
    if abs(bound - e_["target"]) < 0.002:
        near.append((e_["id"], round(m_["point"], 4), round(bound, 4), e_["target"]))
check("6f_decision_robustness", "INFO",
      "endpoints whose deciding bound is within 0.002 of the target (decision sensitive to tiny changes): "
      + (str(near) if near else "none"))

# levels
lv_diff = {"point": 0.0, "se": 0.0}
for nm_, v in INF["levels"].items():
    parts = nm_.split("|")
    if parts[0] == "R":
        k, arm, fmt, w = int(parts[1]), parts[2], parts[3], parts[4]
        pt, rep = R_(k, arm, fmt, w, False), R_(k, arm, fmt, w, True)
    elif parts[0] == "Rrace":
        k, arm, w = int(parts[1]), parts[2], parts[3]
        pt, rep = RR_(k, arm, w, False), RR_(k, arm, w, True)
    elif parts[0] == "acc":
        k, arm, j = int(parts[1]), parts[2], int(parts[3])
        pt, rep = A_(k, arm, j, False), A_(k, arm, j, True)
    elif parts[0] == "Rmean":
        arm, w = parts[1], parts[2]
        pt = mean_list([R_(k, arm, "prim", w, False) for k in SEEDS])
        rep = mean_list([R_(k, arm, "prim", w, True) for k in SEEDS])
    elif parts[0] == "accmean":
        arm, j = parts[1], int(parts[2])
        pt = mean_list([A_(k, arm, j, False) for k in SEEDS])
        rep = mean_list([A_(k, arm, j, True) for k in SEEDS])
    else:
        raise ValueError(nm_)
    lv_diff["point"] = max(lv_diff["point"], abs(float(pt[0]) - v["point"]))
    lv_diff["se"] = max(lv_diff["se"], abs(float(np.std(rep[np.isfinite(rep)], ddof=1)) - v["se"]))
check("6g_levels", "PASS" if max(lv_diff.values()) < 1e-9 else "FAIL",
      f"{len(INF['levels'])} per-seed / mean levels in inference.json (SEX recovery per view and format, race macro AUC, "
      f"deployed accuracy): max |point diff| {lv_diff['point']:.3g}, max |SE diff| {lv_diff['se']:.3g}",
      max_abs_diff=max(lv_diff.values()))
hd = {arm: [round(float(R_(k, arm, "prim", "pair", False)[0] - max(R_(k, arm, "prim", "v1", False)[0],
                                                                    R_(k, arm, "prim", "v2", False)[0])), 4)
            for k in SEEDS] for arm in ("U", "L", "J")}
check("6h_headroom", "INFO", f"outer coalition synergy R_pair - max local per seed: {hd}")

# claims
claims = {}
for claim, arms_needed in ((1, ("J", "L")), (2, ("J", "C*"))):
    ids = [e["id"] for e in LOCK["families"]["primary"] if e["claim"] == claim]
    allpass = all(MINE[i]["decision"] == "PASS" for i in ids)
    noms = all(SL["seeds"][str(k)]["arms"]["J"]["status"] == "NOMINEE" for k in SEEDS)
    if claim == 1:
        noms = noms and all(SL["seeds"][str(k)]["arms"]["L"]["status"] == "NOMINEE" for k in SEEDS)
    else:
        noms = noms and all(cstar(k) is not None for k in SEEDS)
    my_noms = all(MYSEL[k]["arms"]["J"]["status"] == "NOMINEE" for k in SEEDS) and (
        all(MYSEL[k]["arms"]["L"]["status"] == "NOMINEE" for k in SEEDS) if claim == 1 else
        all(MYSEL[k]["cstar"] is not None for k in SEEDS))
    dec = "PASS" if (allpass and noms and my_noms) else "NOT_ESTABLISHED"
    ru = INF[f"claim{claim}"]
    claims[f"claim{claim}"] = {"replay_decision": dec, "runner_decision": ru["decision"], "all_nine_pass": allpass,
                               "nominees_all_seeds": noms and my_noms,
                               "clauses_passing_replay": sum(MINE[i]["decision"] == "PASS" for i in ids),
                               "clauses_passing_runner": ru["clauses_passing"]}
cl_ok = all(c["replay_decision"] == c["runner_decision"] and c["clauses_passing_replay"] == c["clauses_passing_runner"]
            for c in claims.values())
check("6i_claims", "PASS" if cl_ok else "FAIL",
      "; ".join(f"{k_}: replay {v['replay_decision']} (all nine pass={v['all_nine_pass']}, nominees on every seed="
                f"{v['nominees_all_seeds']}, {v['clauses_passing_replay']}/9 clauses) vs runner {v['runner_decision']} "
                f"({v['clauses_passing_runner']}/9)" for k_, v in claims.items()))

# ====================================================================================================== 7 utility table
log("utility table")


def supported_task(j):
    return [c for c in range(KS[j]) if all((Y[j][IDX[r]] == c).sum() >= SUPPORT_MIN
                                           for r in ("attacker_fit", "attacker_val", "assessment"))]


ut = list(csv.DictReader(open(PKG / "ACTUAL_TASK_UTILITY.csv")))
ut_rel, ut_abs, ut_bad, ut_rec = 0.0, 0.0, [], 0.0
for row in ut:
    k, arm, j = int(row["seed"]), row["arm"], 0 if row["task"] == "income" else 1
    p = PR[(k, arm)]
    y = p["y_income"] if j == 0 else p["y_occ"]
    P = p[f"p{j + 1}"]
    h = p[f"hard{j + 1}"]
    sup = supported_task(j)
    rec = {c: float((h[y == c] == c).mean()) for c in range(KS[j]) if (y == c).sum()}
    mino = min(sup, key=lambda c: (y == c).sum())
    conf = P.max(1)
    ece = 0.0
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.sum():
            ece += msk.mean() * abs(conf[msk].mean() - (h[msk] == y[msk]).mean())
    mine = {"accuracy": float((h == y).mean()), "balanced_accuracy_supported": float(np.mean([rec[c] for c in sup])),
            "minority_recall": rec[mino], "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1)))),
            "brier": float(np.mean(((P - np.eye(KS[j])[y]) ** 2).sum(1))), "ece_10bin": float(ece),
            "const_accuracy": float((y == CONST[j]).mean())}
    mine["useful_gain"] = mine["accuracy"] - mine["const_accuracy"]
    if int(row["minority_class"]) != mino or row["status"] != SL["seeds"][str(k)]["arms"][arm]["status"]:
        ut_bad.append(f"s{k}/{arm}/{row['task']}")
    full = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")["utility_deployed"][str(j)]
    for q, v in mine.items():
        ut_rel = max(ut_rel, abs(v - float(row[q])) / max(abs(float(row[q])), 1e-12))
        ut_abs = max(ut_abs, abs(v - float(row[q])))
        ut_rec = max(ut_rec, abs(v - full[q]))
check("7_task_utility_table", "PASS" if (ut_rel < 6e-6 and ut_rec < 1e-12 and not ut_bad and len(ut) == 54) else "FAIL",
      f"{len(ut)} rows (27 outer units x 2 tasks): accuracy, balanced accuracy over supported classes (income "
      f"{supported_task(0)}, occupation {supported_task(1)}), minority recall, log loss, Brier, ECE, constant accuracy "
      f"and useful gain recomputed from preds; max relative diff vs CSV (6 significant digits) {ut_rel:.3g}, max abs diff "
      f"vs outer records {ut_rec:.3g}; minority-class/status disagreements {ut_bad}", max_abs_diff=ut_rec)

nat_rows = list(csv.DictReader(open(PKG / "NATIVE_VS_AUDIT.csv")))
nat_rel, nat_ho = 0.0, 0.0
for row in nat_rows:
    k, arm, i = int(row["seed"]), row["arm"], int(row["recipient"]) - 1
    a = SL["seeds"][str(k)]["arms"][arm]
    if "unit" in a:
        fr, hc = NATIVE_UNIT[a["unit"]][i]
        if fr is not None and row["native_value"]:
            nat_rel = max(nat_rel, abs(fr - float(row["native_value"])))
    else:
        r = np.load(UNITS / a["units"][i] / "release.npz")["r"]
        hc = float(np.abs(colcorr(r[AS], SEX[AS])).max())
    nat_ho = max(nat_ho, abs(hc - float(row["held_out_max_abs_corr_assessment"])) / max(hc, 1e-12))
check("7b_native_vs_audit_table", "PASS" if nat_ho < 6e-6 and nat_rel < 1e-12 else "FAIL",
      f"{len(nat_rows)} rows: LEACE fit-row relative cross-covariance (abs diff {nat_rel:.3g}; both ~1e-14) and held-out "
      f"assessment max |corr(r, SEX)| (max relative diff {nat_ho:.3g}) recomputed from the saved maps/releases",
      max_abs_diff=nat_rel)
fc_rows = [r for r in nat_rows if r["arm"] in ("F", "F0")]
a3 = jload(PKG / "FARE_CERTIFICATES_A3.json") if (PKG / "FARE_CERTIFICATES_A3.json").exists() else None
a3_ok = sum(v["status"] == "OK" for s in (a3 or {"seeds": {}})["seeds"].values() for v in s.values())
check("7c_fare_certificate_reporting", "INFO",
      f"NATIVE_VS_AUDIT.csv lists all {len(fc_rows)} FARE certificate rows as UNAVAILABLE (outer-time refusal: cert rows "
      f"share feature vectors with fit rows); FARE_CERTIFICATES_A3.json (descriptive re-run under a record-identity guard; "
      f"no AMENDMENT_A3 document and not listed in LOCK.json amendments) reports {a3_ok} OK certificates, all vacuous "
      f"(bound > 1). No endpoint uses either.")

# ====================================================================================================== 8 attacker replay
log("attacker replay")


def view_mats(k, arm):
    a = SL["seeds"][str(k)]["arms"][arm]
    return views_fare(*a["units"]) if "units" in a else views_nn(a["unit"])


def finite_arm(arm):
    return arm in ("F", "F0")


SAMPLE = [(0, "J", "primary", "v1"), (0, "L", "primary", "pair"), (0, "F", "primary", "v2"),
          (0, "J", "secondary_hard", "hpair"), (1, "J", "primary", "pair"), (2, "L", "secondary_prob", "p1"),
          (0, "F", "primary", "pair")]
att_rows, att_diff = [], 0.0
for k, arm, sec, w in SAMPLE:
    rec = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")[sec][w]
    V = view_mats(k, arm)
    src = rec["selected_view"]
    Xv = V[src]
    name = rec["selected"]
    P_saved = PR[(k, arm)][f"P_{w}"]
    seeds_done = []
    for s in (0, 1, 2):
        m = est(name, s).fit(Xv[AF], SEX[AF])
        P = proba_k(m, Xv[AS], 2)
        d_ = float(np.abs(P - P_saved[s]).max())
        att_diff = max(att_diff, d_)
        seeds_done.append(d_)
    # also confirm the selection itself on attacker_val among the recorded table (selected = argmin val log loss)
    tab = rec["table"]
    argmin = min(tab, key=lambda t: t["attacker_val_log_loss"])["attacker"]
    m0 = est(name, 0).fit(Xv[AF], SEX[AF])
    vll = ll(SEX[AV], proba_k(m0, Xv[AV], 2))
    att_rows.append(f"s{k}/{arm}/{w}: {name} on view {src}, max|dP| over 3 attacker seeds {max(seeds_done):.3g}, "
                    f"attacker_val log loss diff {abs(vll - rec['val_log_loss']):.3g}"
                    + ("" if (argmin == name or src != w) else f" (table argmin {argmin} != selected)"))
    att_diff = max(att_diff, abs(vll - rec["val_log_loss"]))
# race audit replay (secondary slate on the supported-race subset of the scored roles; K = 5 columns)
KR = int(RACE.max()) + 1
rkeep = np.isin(RACE, rsup)
RAF, RAV = AF[rkeep[AF]], AV[rkeep[AV]]
RAS = AS[rkeep[AS]]
for k, arm, w in ((0, "J", "pair"), (0, "F", "v2"), (1, "L", "v1")):
    rr = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")["race"]["audit"][w]
    V = view_mats(k, arm)
    Xv = V[rr["selected_view"]]
    P_saved = PR[(k, arm)][f"Prace_{w}"]
    ds = []
    for s in (0, 1, 2):
        m = est(rr["selected"], s).fit(Xv[RAF], RACE[RAF])
        ds.append(float(np.abs(proba_k(m, Xv[RAS], KR) - P_saved[s]).max()))
    m0 = est(rr["selected"], 0).fit(Xv[RAF], RACE[RAF])
    dv = abs(ll(RACE[RAV], proba_k(m0, Xv[RAV], KR)) - rr["val_log_loss"])
    att_diff = max(att_diff, max(ds), dv)
    att_rows.append(f"race s{k}/{arm}/{w}: {rr['selected']} on view {rr['selected_view']}, max|dP| {max(ds):.3g}, "
                    f"attacker_val log loss diff {dv:.3g}")
check("8_attacker_replay", "PASS" if att_diff < 1e-9 else "FAIL",
      f"{len(SAMPLE)} SEX + 3 race recorded selections refitted from the attacker name with own estimators (attacker_fit, seeds 0,1,2) "
      "and compared with P_<view> on assessment: " + "; ".join(att_rows), max_abs_diff=att_diff)

# every recorded outer selection equals the attacker_val log-loss argmin (ties -> slate order), coalition bank rule
sel_bad, sel_n, art = [], 0, []
for k in SEEDS:
    for arm in ARMS:
        r = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")
        for sec, ws, ck in (("primary", ("v1", "v2", "pair"), "pair"), ("secondary_prob", ("p1", "p2", "ppair"), "ppair"),
                            ("secondary_hard", ("h1", "h2", "hpair"), "hpair"), ("race", ("v1", "v2", "pair"), "pair")):
            blk = r[sec]["audit"] if sec == "race" else r[sec]
            own = {}
            for w in ws:
                best = None
                for t in blk[w]["table"]:
                    if best is None or t["attacker_val_log_loss"] < best[0] - 1e-12:
                        best = (t["attacker_val_log_loss"], t["attacker"])
                own[w] = best
            for w in ws:
                sel_n += 1
                if w == ck:
                    cands = [(own[w][0], 0, w)] + [(own[o][0], j + 1, o) for j, o in enumerate(x for x in ws if x != ck)]
                    src = min(cands)[2]
                else:
                    src = w
                if w == ck and blk[w]["selected_view"] != w:
                    # outer.py stores the winning LOCAL view's table under the coalition key when an ignore-other-view
                    # attacker wins, so the coalition's own slate table is not in the record
                    lv = blk[w]["selected_view"]
                    if blk[w]["table"] == blk[lv]["table"] and blk[w]["selected"] == own[lv][1] and \
                            blk[w]["val_log_loss"] == own[lv][0] and own[lv][0] <= min(own[o][0] for o in ws if o != ck):
                        art.append(f"s{k}/{arm}/{sec}/{w}->{lv}")
                    else:
                        sel_bad.append(f"s{k}/{arm}/{sec}/{w}")
                elif (blk[w]["selected_view"], blk[w]["selected"], blk[w]["val_log_loss"]) != (src, own[src][1], own[src][0]):
                    sel_bad.append(f"s{k}/{arm}/{sec}/{w}")
        sup_r = r["race"]["supported_classes"]
        if sup_r != rsup or r["race"]["n_assessment"] != int(len(RIDX)):
            sel_bad.append(f"s{k}/{arm}/race support")
        tabs = {len(r["primary"][w]["table"]) for w in ("v1", "v2", "pair")}
        exp_n = 14 + (3 if arm in ("F", "F0") else 0)
        if tabs != {exp_n}:
            sel_bad.append(f"s{k}/{arm}/primary slate size {tabs} != {exp_n}")
nsel_local = sum(jload(UNITS / f"outer__s{k}__{arm}" / "record.json")["primary"]["pair"]["selected_view"] != "pair"
                 for k in SEEDS for arm in ARMS)
check("8b_outer_selection_rule", "PASS" if not sel_bad else "FAIL",
      f"{sel_n} recorded outer attacker selections (primary, output-only prob/hard, race) equal the attacker_val log-loss "
      f"argmin of their own recorded tables (ties -> slate order; coalition bank includes the local selections); primary "
      f"slate has 14 attackers (+3 cell-conditional for FARE); race support {rsup}; coalition picked a local attacker in "
      f"{nsel_local}/27 primary pair views; problems {sel_bad[:10]}")
# refit the coalition's own final slate where a local attacker won, to audit the unrecorded half of the bank decision
FINAL = ([f"LR_C{c}" for c in (0.01, 0.1, 1.0, 10.0, 100.0)] + ["MLP_64", "MLP_128", "MLP_64x64", "MLP_128x128"] +
         [f"HGB_{a}_{b}" for a in (0.05, 0.1) for b in (15, 31)] + ["DA_canonical_MLP"])
cb_rows, cb_ok = [], True
for k, arm in ((2, "J"), (1, "U")):
    r = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")["primary"]["pair"]
    if r["selected_view"] == "pair":
        continue
    Xp = view_mats(k, arm)["pair"]
    best = None
    for nm_ in FINAL:
        m = est(nm_, 0).fit(Xp[AF], SEX[AF])
        l_ = ll(SEX[AV], proba_k(m, Xp[AV], 2))
        if best is None or l_ < best[0] - 1e-12:
            best = (l_, nm_)
    ok_ = r["val_log_loss"] < best[0]
    cb_ok &= ok_
    cb_rows.append(f"s{k}/{arm}: coalition-own best {best[1]} log loss {best[0]:.5f} vs winning local "
                   f"{r['selected_view']}:{r['selected']} {r['val_log_loss']:.5f} -> local wins: {ok_}")
check("8c_coalition_table_recording", "WARN" if art else "PASS",
      f"{len(art)} coalition selections where an ignore-other-view attacker won: the record stores the local view's "
      f"table under the coalition key (outer.audit_views writes sel['table'] of the winning source view), so the "
      f"coalition's own slate losses are absent and the bank decision cannot be audited from records alone. The stored "
      f"values are internally consistent (selected = local argmin, local loss <= other local). Independent refit of the "
      f"coalition's own 14-attacker final slate for 2 primary cases confirms the local attacker wins: "
      + "; ".join(cb_rows) + (" " if cb_ok else " REFIT DISAGREES. ") + f"Cases: {art}")
if not cb_ok:
    CHECKS[-1]["status"] = "FAIL"

# ====================================================================================================== 9 controls
ctl = jload(RUN / "controls.json")
pub = jload(PKG / "AUDIT_CONTROLS.json") if (PKG / "AUDIT_CONTROLS.json").exists() else None
nulls = [v["null_val_auc"] for v in ctl.values()]
pl = [v["planted_val_auc"] for v in ctl.values()]
ok9 = max(nulls) <= 0.55 and min(pl) > 0.75 and (pub is None or pub == ctl)
check("9_controls", "PASS" if ok9 else "FAIL",
      f"{len(ctl)} views (J, L, F seed 0; v1/v2/pair): label-permutation null AUC max {max(nulls):.4f} (<= 0.55), planted "
      f"leak AUC min {min(pl):.4f} (> 0.75); public AUDIT_CONTROLS.json identical to private controls.json: {pub == ctl}")

# ====================================================================================================== 10 integrity
log("integrity")
by_type, bad_hash, extra_files, nq = {}, [], [], 0
for d in sorted(UNITS.iterdir()):
    if not d.is_dir():
        continue
    if d.name.endswith(".quarantined"):
        nq += 1
        t = "quarantined:" + d.name.split("__")[0]
    elif d.name.startswith("inner__"):
        t = "inner__" + d.name.split("__")[1]
    elif d.name.startswith("fare__"):
        t = "fare_Z" if "__Z" in d.name else "fare_c"
    else:
        t = d.name.split("__")[0]
    by_type[t] = by_type.get(t, 0) + 1
    cj = d / "COMPLETE.json"
    if not cj.exists():
        bad_hash.append(f"{d.name}: no COMPLETE.json")
        continue
    files = jload(cj)["files"]
    for f, h in files.items():
        p = d / f
        if not p.exists() or sha_file(p) != h:
            bad_hash.append(f"{d.name}/{f}")
    actual = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    if actual != set(files):
        extra_files.append(d.name)
check("10a_complete_json_hashes", "PASS" if not bad_hash and not extra_files else "FAIL",
      f"{sum(by_type.values())} unit directories ({nq} quarantined) - every file hash in COMPLETE.json verifies and no "
      f"unlisted files: bad {bad_hash[:10]}, unlisted {extra_files[:10]}")
active = {k_: v for k_, v in by_type.items() if not k_.startswith("quarantined")}
exp_types = {"warm": 3, "nn": 51, "fare_c": 36, "fare_Z": 6, "inner__nn": 51, "inner__fare": 36, "inner__pair": 6,
             "outer": 27}
check("10b_unit_counts", "PASS" if active == exp_types and sum(active.values()) == 216 else "FAIL",
      f"active units by type {active} (total {sum(active.values())}; UNIT_MANIFEST plans 216); quarantined "
      f"{ {k_: v for k_, v in by_type.items() if k_.startswith('quarantined')} }")

# quarantined vs current: only output arrays differ
q_bad, q_cdiff, q_nonfinite, q_n = [], 0.0, 0, 0
q_pdiff, q_pchanged, q_parrays, nf_where = 0.0, 0, 0, []
nf_rows_per_unit = sorted({int((~np.isfinite(np.load(UNITS / f"nn__s{k}__U.quarantined" / "release.npz")["c1"])).any(1).sum())
                           for k in SEEDS})
for qd in sorted(UNITS.glob("*.quarantined")):
    cur = UNITS / qd.name[:-len(".quarantined")]
    qa, ca = jload(qd / "COMPLETE.json")["files"], jload(cur / "COMPLETE.json")["files"]
    if set(qa) != set(ca):
        q_bad.append(f"{cur.name}: file lists differ")
    for f in qa:
        if f in ("release.npz", "record.json"):
            continue
        if qa[f] != ca.get(f):
            q_bad.append(f"{cur.name}/{f}")
    rq, rc = jload(qd / "record.json"), jload(cur / "record.json")
    if {k_: v for k_, v in rc.items() if k_ != "amendment_A1"} != rq or "amendment_A1" not in rc:
        q_bad.append(f"{cur.name}/record.json beyond amendment_A1")
    zq, zc = np.load(qd / "release.npz"), np.load(cur / "release.npz")
    if set(zq.files) != set(zc.files):
        q_bad.append(f"{cur.name}: release keys")
    for f in zq.files:
        if f in ("c", "c1", "c2"):
            a_, b_ = zq[f], zc[f]
            fin = np.isfinite(a_)
            nf = int((~fin).sum())
            q_nonfinite += nf
            if nf:
                nf_where.append(f"{cur.name}/{f}:{int((~fin).any(1).sum())} rows")
            q_cdiff = max(q_cdiff, float(np.abs(a_[fin] - b_[fin]).max()) if fin.any() else 0.0)
            if not np.isfinite(b_).all():
                q_bad.append(f"{cur.name}/{f}: current nonfinite")
        elif f in ("p", "p1", "p2"):
            q_pdiff = max(q_pdiff, float(np.abs(zq[f] - zc[f]).max()))
            q_pchanged += int(not np.array_equal(zq[f], zc[f]))
            q_parrays += 1
        elif not np.array_equal(zq[f], zc[f]):
            q_bad.append(f"{cur.name}/release:{f}")
    q_n += 1
check("10c_quarantined_vs_current", "PASS" if not q_bad and q_pdiff < 1e-15 else "FAIL",
      f"{q_n} quarantined (pre-A1) units vs current: model.pt, LEACE maps and heads identical (hash); record differs only "
      f"by the amendment_A1 key; in release.npz r, hard, cells and row ids are bitwise equal and only output arrays "
      f"differ: centred logits (pre-A1 had {q_nonfinite} non-finite entries, in {nf_where}; finite entries differ by "
      f"<= {q_cdiff:.3g}) and probabilities (<= {q_pdiff:.3g}). Problems {q_bad[:10]}", max_abs_diff=max(q_cdiff, q_pdiff))
check("10c2_A1_probability_wording", "WARN" if q_pchanged else "PASS",
      f"AMENDMENT_A1 says probabilities are unchanged, but {q_pchanged}/{q_parrays} probability arrays changed at round-off "
      f"level (max {q_pdiff:.3g}). Root cause: the locked pre-A1 finalize.outputs computed P = exp(predict_log_proba); "
      f"A1 computes P = predict_proba. Hard decisions are bitwise unchanged (the runner asserts this), every inner, "
      f"selection and outer result used the post-A1 arrays, so there is no decision effect; the amendment text is "
      f"imprecise, not the data. Also: A1 says 'income, 390 rows each'; the quarantined U units have "
      f"{nf_rows_per_unit} rows per seed with a zero income probability, i.e. 390 non-finite ENTRIES (2 per row) per seed.")

sa_bad, sa_n = [], 0
for k in SEEDS:
    for arm, a in SL["seeds"][str(k)]["arms"].items():
        for unit, files in (a.get("artifact_sha256") or {}).items():
            for f, h in files.items():
                sa_n += 1
                if sha_file(UNITS / unit / f) != h:
                    sa_bad.append(f"{unit}/{f}")
check("10d_selection_lock_artifact_hashes", "PASS" if not sa_bad and sa_n > 0 else "FAIL",
      f"{sa_n} artifact hashes recorded in SELECTION_LOCK.json vs current files: mismatches {sa_bad}")

os_bad = []
for k in SEEDS:
    for arm in ARMS:
        r = jload(UNITS / f"outer__s{k}__{arm}" / "record.json")
        a = SL["seeds"][str(k)]["arms"][arm]
        if r["source"] != a.get("unit", a.get("units")) or r["status"] != a["status"] or r.get("beta") != a.get("beta"):
            os_bad.append(f"s{k}/{arm}")
check("10f_outer_source_units", "PASS" if not os_bad else "FAIL",
      f"27 outer records: source unit(s), status and beta equal SELECTION_LOCK (bad {os_bad})")

cf_bad = [f for f, h in LOCK["code_files"].items() if not (WT / f).exists() or sha_file(WT / f) != h]
check("10g_lock_code_hashes", "PASS" if not cf_bad else "FAIL",
      f"{len(LOCK['code_files']) - len(cf_bad)}/{len(LOCK['code_files'])} LOCK.json code_files match the working tree "
      f"(mismatches {cf_bad})")
doc_bad = [f for f, h in LOCK["documents_sha256"].items() if not (PKG / f).exists() or sha_file(PKG / f) != h]
doc_note = ""
if "UNIT_MANIFEST.csv" in doc_bad:
    rel_ = "results/pcrl_joint_complete_view_method_v1/UNIT_MANIFEST.csv"
    old = subprocess.run(["git", "-C", str(WT), "show", f"a984a1e:{rel_}"], capture_output=True).stdout
    if hashlib.sha256(old).hexdigest() == LOCK["documents_sha256"]["UNIT_MANIFEST.csv"]:
        o_rows = list(csv.reader(old.decode().splitlines()))
        n_rows = list(csv.reader((PKG / "UNIT_MANIFEST.csv").read_text().splitlines()))
        same_plan = sum(a[:6] == b[:6] for a, b in zip(o_rows, n_rows))
        ph = sum(a[:6] != b[:6] and a[0].replace("<F-selected>", "1") == b[0] and a[1:6] == b[1:6]
                 for a, b in zip(o_rows, n_rows))
        st_only = sum(a[6] != b[6] for a, b in zip(o_rows, n_rows))
        other = len(o_rows) - same_plan - ph
        last = subprocess.run(["git", "-C", str(WT), "log", "-1", "--format=%h %cI", "--", rel_], capture_output=True,
                              text=True).stdout.strip()
        doc_note = (f" UNIT_MANIFEST.csv: the locked version (a984a1e) matches the LOCK hash; the current file (commit "
                    f"{last}) has {len(n_rows)} rows vs {len(o_rows)}, planned columns identical in {same_plan} rows, the "
                    f"F0 placeholder 'Z<F-selected>' resolved to 'Z1' in {ph} rows, status column changed in {st_only} rows "
                    f"(planned -> complete), other differences {other}. Post-lock bookkeeping, not a plan change, but the "
                    f"locked document hash no longer verifies.")
check("10h_lock_document_hashes", "PASS" if not doc_bad else "WARN",
      f"{len(LOCK['documents_sha256']) - len(doc_bad)}/{len(LOCK['documents_sha256'])} locked protocol documents unchanged "
      f"(changed: {doc_bad}).{doc_note}")


def git(*args):
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True).stdout.strip()


SL_COMMIT = "197f323be81d41e0caef12dca8761bd32f512f78"
remote = git("branch", "-r", "--contains", SL_COMMIT)
sl_changed = git("diff", "--stat", SL_COMMIT, "--", "results/pcrl_joint_complete_view_method_v1/SELECTION_LOCK.json")
last_sl = git("log", "-1", "--format=%H", "--", "results/pcrl_joint_complete_view_method_v1/SELECTION_LOCK.json")
ct = git("show", "-s", "--format=%cI", SL_COMMIT)
check("10i_selection_lock_pushed_and_unchanged",
      "PASS" if "origin/research/pcrl-joint-complete-view-method-v1" in remote and not sl_changed and last_sl == SL_COMMIT
      else "FAIL",
      f"{SL_COMMIT[:9]} is on remote branches [{remote.strip()}]; last commit touching SELECTION_LOCK.json = "
      f"{last_sl[:9]}; working-tree diff vs lock commit: {'none' if not sl_changed else sl_changed}")
acts = [json.loads(line) for line in open(RUN / "ACTIVITY_LOG.jsonl")]
outer_t = sorted(a["at"] for a in acts if a.get("unit", "").startswith("outer__"))
sel_end = max(a["at"] for a in acts if a["event"] == "end select")
import datetime as _dt  # noqa: E402
ct_utc = _dt.datetime.fromisoformat(ct).astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
check("10j_timeline", "PASS" if outer_t and outer_t[0] > ct_utc > sel_end else "FAIL",
      f"selection ended {sel_end}; SELECTION_LOCK commit {ct_utc}; first outer unit completed {outer_t[0]}, last "
      f"{outer_t[-1]} ({len(outer_t)} outer completions); commit time is a lower bound on push time (outer.py refuses "
      f"to run unless the lock commit is on the remote)")

A2_COMMIT = "a984a1e421fa1c586ee76f488322c2ab5bb404e5"
a2t = _dt.datetime.fromisoformat(git("show", "-s", "--format=%cI", A2_COMMIT)).astimezone(_dt.timezone.utc) \
    .strftime("%Y-%m-%dT%H:%M:%SZ")
done_at = {a["unit"]: a["at"] for a in acts if a.get("unit", "").startswith("outer__")}
pre = sorted(u for u, t in done_at.items() if t < a2t)
post = sorted(u for u, t in done_at.items() if t >= a2t)
exp_pre = sorted(f"outer__s{k}__{a}" for k in (0, 2) for a in ("U", "E", "L", "J", "JP", "S12", "S21"))
check("10k_A2_scope", "PASS" if pre == exp_pre and len(post) == 13 else "FAIL",
      f"A2 commit {a2t}: {len(pre)} outer units completed before it (expected the 14 seed-0/2 neural units: "
      f"{pre == exp_pre}); {len(post)} after (F/F0 seeds 0/2 and all of seed 1). The pre-A2 units ran the cell-conditional "
      f"attacker only on binary SEX (hard views), where the A2 change (K = max(K, max(y)+1)) is a no-op")
eu = [sha_file(UNITS / uname(k, "E") / "model.pt") == sha_file(UNITS / uname(k, "U") / "model.pt") for k in SEEDS]
check("10l_E_uses_U_encoders", "PASS" if all(eu) else "FAIL",
      f"E's model.pt is byte-identical to U's on seeds {SEEDS}: {eu} (E = U encoders + final LEACE + refitted heads)")

# ====================================================================================================== finish
loaded = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
assert not loaded, f"forbidden modules loaded: {loaded}"
check("0_import_guard", "PASS" if not loaded and not _ImportGuard.blocked else "FAIL",
      f"meta_path guard active for {list(FORBIDDEN)}; forbidden modules loaded at end: {loaded}; blocked attempts: "
      f"{_ImportGuard.blocked}")

summary = {s: sum(c["status"] == s for c in CHECKS) for s in ("PASS", "FAIL", "WARN", "INFO")}
summary["total"] = len(CHECKS)
summary["max_abs_diff"] = {
    "deployment_neural": mx, "deployment_fare": max(fdep["c"], fdep["p"]),
    "leace_native_fit_rows_rel": native["fit_rel_max"], "inner_utility": iu_diff, "selection_values": val_diff,
    "inner_recovery_spot": ir_diff, "endpoint_points": agg["point"], "endpoint_ses": agg["se"],
    "levels_points": lv_diff["point"], "levels_ses": lv_diff["se"], "utility_vs_records": ut_rec,
    "attacker_replay": att_diff}
out = {
    "schema": "jcv-independent-verification-v1",
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "replay_script": "results/pcrl_joint_complete_view_method_v1/verification/replay_jcv.py",
    "replay_script_sha256": sha_file(Path(__file__)),
    "inputs": {"private_inputs": tilde(INPUTS), "private_inputs_sha256": ish, "units_dir": tilde(UNITS),
               "selection_lock_commit": SL_COMMIT, "worktree_head": git("rev-parse", "HEAD")},
    "forbidden_imports": list(FORBIDDEN),
    "import_guard": True,
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                    "sklearn": __import__("sklearn").__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
    "bootstrap": {"B": B, "seed": BSEED, "n_assessment": n, "n_groups": NU, "z_primary": ZP, "z_secondary": ZS,
                  "distinct_base_statistics": len(BASE)},
    "checks": CHECKS,
    "primary": COMP["primary"],
    "secondary": COMP["secondary"],
    "claims": claims,
    "selection_replay": {str(k): {"C*": MYSEL[k]["cstar"],
                                  **{arm: {"status": MYSEL[k]["arms"][arm]["status"],
                                           "unit": MYSEL[k]["arms"][arm].get("unit", MYSEL[k]["arms"][arm].get("units")),
                                           "gates_ok": MYSEL[k]["arms"][arm]["gates_ok"],
                                           "worst_gate_margin": MYSEL[k]["arms"][arm]["margin"]} for arm in ARMS}}
                         for k in SEEDS},
    "summary": summary,
    "wall_s": round(time.time() - T0, 1),
}
txt = json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
assert "/Users/" not in txt, "absolute user path in output"
OUT.write_text(txt + "\n")
log("summary", summary)
log(f"wrote {OUT.relative_to(WT)}")
