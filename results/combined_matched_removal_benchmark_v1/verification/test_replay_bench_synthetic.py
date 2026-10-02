#!/Users/nathansamson/PCRL/.venv/bin/python
"""Synthetic validation of replay_bench.py (ROLE 4), before any real benchmark output is read.

Builds, in a scratch directory, a private root in the BENCH_DESIGN layout
(inputs/INPUTS_INDEX.json + record keys / roles / labels / forward caches; units/<unit_id>/{preds.npz,
supported.json, fit_records.json, COMPLETE.json[, ALIAS.json]}; defenses/<unit_id>/{map.npz, meta.json,
transformed.npz}) with known ground truth:

  * 2 datasets (adult: target sex, policy {race, sex}; hmda: target race, policy {race, ethnicity}),
    2 encoder seeds, arms A / B / C / D (2 sigmas x 2 release seeds), attacker seeds {0, 1, 2};
  * official concept-erasure LeaceFitter maps (torch) for B and C, fit on train-split rows whose record key
    does not occur in the test split (80 shared-key train rows must be excluded);
  * aliased B == C: hmda ethnicity is a deterministic function of race, so the policy concept spans the same
    space and C's map equals B's -> C is scored once and reported as alias rows;
  * NE class: hmda race class 3 (about 1 %) fails 100/30/100 -> macro over {0, 1, 2, 4};
  * near-tied worst class: hmda untreated rep attacker with per-class signal (0.40, 0.45, 0.45, -, 0.44);
  * null: adult B rep attacker carries no signal (Rrep BELOW expected), adult C borderline (true AUC 0.55);
  * duplicates: 120 test records appear twice (one cluster unit, one role);
  * ignore-rep / ignore-outputs candidates in the C_rep_plus_clean_out slate; deterministic L aliased over seeds;
  * sigma*: adult val AUC <= 0.55 at sigma 0.5 (sigma* = 0.5); hmda never qualifies -> sigma* = max grid, flagged;
  * U2 non-inferiority: B exact copy (NONINFERIOR), adult C 15 % swapped rows (INFERIOR), adult D ~ -0.01 (UNRESOLVED).

A reference implementation inside this file (sklearn roc_auc_score / log_loss, scipy rankdata, index-resampling
cluster bootstrap with numpy RandomState) produces the ground truth and a synthetic runner report; replay_bench
must agree within the frozen tolerances.  Mutation runs check that the replay detects defects.  A null-calibration
simulation (>= 200 replicates) reports the empirical coverage of the derived worst-class bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from concept_erasure import LeaceFitter
from scipy.stats import norm, rankdata
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_bench as RB  # noqa: E402  (module under test; numpy only, not a repo import)

DEFAULT_WORK = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/"
                    "scratchpad/bench_verify")
SIGMAS, ENC, RS, AS = (0.5, 2.0), (0, 1), (0, 1), (0, 1, 2)
SALTS = {"adult": "pilot-roles-v1|", "hmda": "bench-roles-hmda-v1|"}
CELLS = RB.CELLS
D = 10
SPEC = {
    "adult": dict(n_test=4500, n_dup=120, n_train=3500, n_shared=80,
                  attrs={"sex": [0.65, 0.35], "race": [0.55, 0.2, 0.12, 0.05, 0.08]}),
    "hmda": dict(n_test=5000, n_dup=120, n_train=3500, n_shared=80,
                 attrs={"race": [0.5, 0.25, 0.17, 0.04, 0.04]}),
}
# attacker signal per (dataset, group, surface): scalar or per-class vector (GBT strength; MLP 0.9x, L 0.8x)
SIG = {
    ("adult", "A", "rep"): 1.2, ("adult", "B", "rep"): 0.0, ("adult", "C", "rep"): float(norm.ppf(0.55)),
    ("adult", "D_sigma0.5", "rep"): 0.05, ("adult", "D_sigma2", "rep"): 0.0,
    ("adult", "A", "repPLUSoutputs"): 1.4, ("adult", "B", "repPLUSoutputs"): 0.5,
    ("adult", "C", "repPLUSoutputs"): 0.5, ("adult", "D_sigma0.5", "repPLUSoutputs"): 0.6,
    ("adult", "D_sigma2", "repPLUSoutputs"): 0.55, ("adult", "*", "outputs"): 0.45,
    ("hmda", "A", "rep"): [0.40, 0.45, 0.44, 0.0, 0.0], ("hmda", "B", "rep"): 0.0,
    ("hmda", "D_sigma0.5", "rep"): 0.35, ("hmda", "D_sigma2", "rep"): 0.3,
    ("hmda", "A", "repPLUSoutputs"): 0.6, ("hmda", "B", "repPLUSoutputs"): 0.3,
    ("hmda", "D_sigma0.5", "repPLUSoutputs"): 0.4, ("hmda", "D_sigma2", "repPLUSoutputs"): 0.35,
    ("hmda", "*", "outputs"): 0.3,
}
U2_SWAP = {("adult", "B"): 0.0, ("adult", "C"): 0.15, ("adult", "D_sigma0.5"): 0.0167, ("adult", "D_sigma2"): 0.05,
           ("hmda", "B"): 0.0, ("hmda", "D_sigma0.5"): 0.0, ("hmda", "D_sigma2"): 0.0}


def own_role(salt, key):   # separately typed copy of the frozen role rule
    u = int(hashlib.sha256(f"{salt}{key}".encode()).hexdigest()[:8], 16) / 4294967296.0
    return "attacker_fit" if u < 0.5 else ("attacker_val" if u < 0.65 else "assessment")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def srng(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def fs(x):
    return RB.fmt_sigma(x)


def own_support(y, role, need_def=False):
    """Scoring support per role (coordinator decision 2026-10-02): 100 / 30 / 100, the same for every arm."""
    out = []
    for c in sorted(np.unique(y[np.isin(role, ["attacker_fit", "attacker_val", "assessment"])])):
        ok = (np.sum((y == c) & (role == "attacker_fit")) >= 100 and np.sum((y == c) & (role == "attacker_val")) >= 30
              and np.sum((y == c) & (role == "assessment")) >= 100)
        if ok:
            out.append(int(c))
    return out


# ================================================================ generation
def gen_dataset(name, work, rng):
    """Inputs in the admission schema (bench_v1.inputs_index/v1)."""
    sp = SPEC[name]
    inp = work / "inputs"
    cell = CELLS[name]
    nt = sp["n_test"]
    canon = np.array([rng.bytes(10).hex() for _ in range(nt + sp["n_train"] - sp["n_shared"])])
    nkeys = len(canon)
    rec = {}
    for a, pr in sp["attrs"].items():
        rec[a] = rng.choice(len(pr), size=nkeys, p=pr)
    if name == "adult":
        rec["income"] = (rng.random(nkeys) < 1 / (1 + np.exp(-(-1.0 + 0.9 * rec["sex"])))).astype(int)
    else:
        rec["ethnicity"] = np.isin(rec["race"], [1, 3]).astype(int)          # deterministic in race -> B == C alias
        rec["loan_decision"] = (rng.random(nkeys) < 1 / (1 + np.exp(-(0.6 - 0.5 * (rec["race"] == 1))))).astype(int)
    latent = rng.normal(size=(nkeys, D))
    dup = rng.choice(nt, size=sp["n_dup"], replace=False)
    shared = rng.choice(nt, size=sp["n_shared"], replace=False)
    test_rec = rng.permutation(np.r_[np.arange(nt), dup])
    train_rec = rng.permutation(np.r_[np.arange(nt, nkeys), shared])
    row_rec = np.r_[test_rec, train_rec]
    split = np.array(["test"] * len(test_rec) + ["train"] * len(train_rec))
    N = len(row_rec)
    row_id = np.arange(N, dtype=np.int64)                        # test 0..n_test-1, train n_test + position
    ck = canon[row_rec]
    # Adult: the pilot record_key of a test row hashes the raw row ('>50K.'), canon_key strips the '.'
    rk = np.array([hashlib.sha256((c + ".").encode()).hexdigest()[:20] if (name == "adult" and s_ == "test") else c
                   for c, s_ in zip(ck, split)])
    unit_of = dict(zip(canon, rng.permutation(nkeys) + 7))
    unit = np.array([unit_of[k] for k in ck], dtype=np.int64)
    test_canon = set(ck[split == "test"])
    role = np.array([own_role(SALTS[name], r) if s_ == "test" else ("excluded_dup" if c in test_canon else "defense_fit")
                     for r, c, s_ in zip(rk, ck, split)])
    labels = {a: rec[a][row_rec] for a in rec}
    tcol = f"task_{cell['task']}"
    lab_np = {"row_id": row_id, "split": split, "unit": unit, "record_key": rk, "canon_key": ck, "role": role,
              **{a: v for a, v in labels.items()}, tcol: labels[cell["task"]]}
    np.savez(inp / f"{name}_labels.npz", **lab_np)
    roles_np, roles_meta = {}, {}
    for r in ("defense_fit", "attacker_fit", "attacker_val", "assessment", "excluded_dup"):
        m = role == r
        roles_np[f"{r}__row_id"] = row_id[m]
        roles_np[f"{r}__unit"] = unit[m]
        roles_meta[r] = {"n_rows": int(m.sum()), "n_units": int(len(np.unique(unit[m]))),
                         "row_id_sha256": hashlib.sha256(row_id[m].astype("<i8").tobytes()).hexdigest(),
                         "unit_sha256": hashlib.sha256(unit[m].astype("<i8").tobytes()).hexdigest(),
                         "splits": sorted(set(split[m]))}
    np.savez(inp / f"{name}_roles.npz", **roles_np)
    encs, H_by_seed, logit_by_seed = {}, {}, {}
    for s in ENC:
        r = srng(name, "enc", s)
        X = latent[row_rec].copy()
        if name == "adult":
            X[:, 0] += 1.5 * labels["sex"]
            X[:, 1] += 0.8 * labels["income"]
            if s == 0:                     # seed 1 carries no race signal -> C truncates sub-tolerance directions
                X[:, 2:7] += 0.6 * np.eye(5)[labels["race"]]
        else:
            X[:, :5] += 0.9 * np.eye(5)[labels["race"]]
            X[:, 5] += 0.8 * labels["loan_decision"]
        Q, _ = np.linalg.qr(r.normal(size=(D, D)))
        H = (X @ Q) * r.uniform(0.5, 2.0, size=D) + r.normal(size=D)
        if name == "hmda" and s == 0:      # a dead unit: rank-deficient defense_fit sample covariance
            H[:, D - 1] = 0.37
        Lg = H @ r.normal(size=(D, 2))
        o3 = np.argsort(-row_id)       # cache in descending row_id order (alignment is by row_id)
        fwd = inp / f"{name}_s{s}_forward.npz"
        np.savez(fwd, row_id=row_id[o3], split=split[o3], rep_p0=H[o3], **{f"logits_{cell['purpose']}": Lg[o3]})
        encs[str(s)] = {"checkpoint": f"synthetic/{name}_s{s}_final.pt", "checkpoint_sha256": "0" * 64,
                        "lineage_status": "ADMITTED", "forward_npz": str(fwd), "forward_sha256": sha(fwd)}
        H_by_seed[s], logit_by_seed[s] = H, Lg
    dims = {a: len(SPEC[name]["attrs"][a]) if a in SPEC[name]["attrs"] else 2 for a in cell["policy"]}
    spec = {"defense_route": "PREFERRED",
            "labels_npz": str(inp / f"{name}_labels.npz"), "labels_sha256": sha(inp / f"{name}_labels.npz"),
            "roles_npz": str(inp / f"{name}_roles.npz"), "roles_sha256": sha(inp / f"{name}_roles.npz"),
            "record_key_array": {"file": str(inp / f"{name}_labels.npz"), "key": "record_key", "canon_key": "canon_key",
                                 "sha256_record_key": hashlib.sha256("\n".join(rk.tolist()).encode()).hexdigest(),
                                 "sha256_canon_key": hashlib.sha256("\n".join(ck.tolist()).encode()).hexdigest()},
            "role_array": {"file": str(inp / f"{name}_labels.npz"), "key": "role", "unit_key": "unit",
                           "sha256": hashlib.sha256("\n".join(role.tolist()).encode()).hexdigest()},
            "roles": roles_meta,
            "purposes": {cell["purpose"]: {"index": 0, "task": cell["task"], "task_dim": 2,
                                           "disallowed_attrs": list(cell["policy"]), "disallowed_attr_dims": dims,
                                           "rep_key": "rep_p0", "logits_key": f"logits_{cell['purpose']}",
                                           "labels_task_key": tcol}},
            "tier1": {"purpose": cell["purpose"], "task": cell["task"], "target": cell["target"],
                      "policy": list(cell["policy"]), "purpose_index": 0},
            "encoders": encs}
    return spec, {"row_id": row_id, "role": role, "unit": unit, "labels": labels, "H": H_by_seed,
                  "logits": logit_by_seed, "rkey": rk, "ckey": ck, "split": split, "dims": dims}


def fit_leace(Hf, Z):
    f = LeaceFitter.fit(torch.from_numpy(Hf), torch.from_numpy(Z))
    e = f.eraser
    return e.proj_left.numpy(), e.proj_right.numpy(), e.bias.numpy(), e


def save_map(dd, Hf, Z):
    """Official fit; saved in the coordinator's layout: proj_left, proj_right, mean_x, mean_z, sigma_xx_used,
    sigma_xz, singular_values (npz) + rank / settings (json, written by the caller)."""
    f = LeaceFitter.fit(torch.from_numpy(Hf), torch.from_numpy(Z))
    e = f.eraser
    sig, sxz = f.sigma_xx, f.sigma_xz
    L, V = torch.linalg.eigh(sig)
    mask = L > (L[-1] * sig.shape[-1] * torch.finfo(L.dtype).eps)
    W = V * torch.where(mask, L.clamp_min(0).rsqrt(), 0.0) @ V.mH
    sv = torch.linalg.svdvals(W @ sxz)
    np.savez(dd / "map.npz", proj_left=e.proj_left.numpy(), proj_right=e.proj_right.numpy(), mean_x=e.bias.numpy(),
             mean_z=f.mean_z.numpy(), sigma_xx_used=sig.numpy(), sigma_xz=sxz.numpy(), singular_values=sv.numpy())
    return e.proj_left.numpy(), e.proj_right.numpy(), e.bias.numpy(), e, sv.numpy()


def onehot(y, classes):
    return (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(np.float64)


def make_scores(rng, y, K, a):
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), (K,)) if np.ndim(a) else np.full(K, float(a))
    z = rng.normal(size=(len(y), K))
    z[np.arange(len(y)), y] += a[y]
    return softmax(z)


def gen_units(name, work, G):
    cell = CELLS[name]
    root_u, root_d = work / "units", work / "defenses"
    lab, role, rid = G["labels"], G["role"], G["row_id"]
    y, yt = lab[cell["target"]], lab[cell["task"]]
    K = int(y.max()) + 1
    rng = srng(name, "rows")
    a_rows = rng.permutation(np.flatnonzero(role == "assessment"))
    v_rows = rng.permutation(np.flatnonzero(role == "attacker_val"))
    fm, dfm = role == "attacker_fit", role == "defense_fit"
    prior = np.array([np.mean(y[fm] == c) for c in range(K)])
    truth = {"alias": {}, "maps": {}}
    for s in ENC:
        H = G["H"][s]
        reps = {"A": H}
        for arm in ("B", "C"):
            attrs = [cell["target"]] if arm == "B" else list(cell["policy"])
            cls = {a: list(range(G["dims"][a])) for a in attrs}          # full declared one-hot
            Z = np.concatenate([onehot(lab[a], cls[a]) for a in attrs], axis=1)
            mid = f"{name}__s{s}__{cell['purpose']}__{cell['target']}__{arm}"
            dd = root_d / mid
            dd.mkdir(parents=True, exist_ok=True)
            Lm, Rm, b, er, sv = save_map(dd, H[dfm], Z[dfm])
            Xe = er(torch.from_numpy(H)).numpy()
            reps[arm] = Xe
            te = np.flatnonzero(np.isin(role, ["attacker_fit", "attacker_val", "assessment"]))
            np.savez(dd / "transformed.npz", row_id=rid[te], H=Xe[te])
            xc = (Xe[dfm] - Xe[dfm].mean(0)).T @ (Z[dfm] - Z[dfm].mean(0)) / (dfm.sum() - 1)
            meta = {"concept_spec": {"attributes": attrs, "classes": cls}, "rank": int(np.sum(sv > 0.01)),
                    "settings": {"svd_tol": 0.01, "shrinkage": True, "method": "leace", "affine": True,
                                 "constrain_cov_trace": True},
                    "fit_row_ids_sha256": hashlib.sha256(np.sort(rid[dfm]).astype(np.int64).tobytes()).hexdigest(),
                    "native_check": {"xcov_maxabs": float(np.max(np.abs(xc)))}}
            truth["maps"][mid] = {"P": np.eye(D) - Lm @ Rm, "b": b, "Xe": Xe, "scale": max(1.0, np.max(np.abs(H[dfm])))}
            (dd / "meta.json").write_text(json.dumps(meta))
        bid = f"{name}__s{s}__{cell['purpose']}__{cell['target']}__B"
        cid = bid[:-1] + "C"
        tb, tc = truth["maps"][bid], truth["maps"][cid]
        alias = np.max(np.abs(tb["Xe"] - tc["Xe"])) / tb["scale"] <= 1e-10      # same action on every row
        truth["alias"][s] = bool(alias)
        if alias:
            m = json.loads((root_d / cid / "meta.json").read_text())
            m["alias_of"] = bid
            (root_d / cid / "meta.json").write_text(json.dumps(m))
        for sg in SIGMAS:
            for k in RS:
                rid_cache = np.sort(rid)[::-1]
                pos = {int(r): i for i, r in enumerate(rid)}
                noise = np.random.default_rng(k).normal(0.0, sg, size=H.shape)
                nl = np.zeros_like(H)
                nl[[pos[int(r)] for r in rid_cache]] = noise
                reps[f"D_sigma{fs(sg)}_rs{k}"] = H + nl
        whead = srng(name, "head", s).normal(size=(D, 2))
        U2_A = None
        for arm_key, Hm in reps.items():
            grp = arm_key.split("_rs")[0]
            uid = f"{name}__s{s}__{cell['purpose']}__{cell['target']}__{arm_key}"
            udir = root_u / uid
            udir.mkdir(parents=True, exist_ok=True)
            sup = own_support(y, role)
            supj = {"sensitive": {"supported_classes": sup,
                                  "supported_pairs": [[a, b] for i, a in enumerate(sup) for b in sup[i + 1:]]}}
            (udir / "supported.json").write_text(json.dumps(supj))
            (udir / "fit_records.json").write_text(json.dumps({"unit": uid}))
            if arm_key == "C" and truth["alias"][s]:
                (udir / "ALIAS.json").write_text(json.dumps({"alias_of": bid}))
                (udir / "COMPLETE.json").write_text(json.dumps({"files": {
                    f: sha(udir / f) for f in ("supported.json", "fit_records.json", "ALIAS.json")}}))
                continue
            pr = {"assess_row_id": rid[a_rows], "assess_unit": G["unit"][a_rows], "y_s": y[a_rows],
                  "y_task": yt[a_rows], "val_row_id": rid[v_rows], "s_prior_fit": prior,
                  "t_prior_fit": np.array([np.mean(yt[fm] == c) for c in range(2)])}
            # attacker slates
            sel = {}
            for surf in ("rep", "outputs", "repPLUSoutputs"):
                a = SIG[(name, "*", "outputs")] if surf == "outputs" else SIG[(name, grp, surf)]
                cand = {}
                for fam, mult in (("L", 0.8), ("GBT", 1.0), ("MLP", 0.9)):
                    seedtag = (name, s, "outputs", fam) if surf == "outputs" else (name, s, arm_key, surf, fam)
                    for ak in AS:
                        if fam == "L" and ak > 0:
                            pr[f"P__{surf}__L__as{ak}"] = pr[f"P__{surf}__L__as0"]
                            continue
                        r = srng(*seedtag, ak)
                        pr[f"P__{surf}__{fam}__as{ak}"] = make_scores(r, y[a_rows], K, np.asarray(a) * mult)
                        if ak == 0:
                            pr[f"V__{surf}__{fam}__as0"] = make_scores(srng(*seedtag, "val"), y[v_rows], K,
                                                                     np.asarray(a) * mult)
                    cand[fam] = fam
                if surf == "repPLUSoutputs":
                    for cnd, src in (("ignore_rep", "outputs"), ("ignore_out", "rep")):
                        pr[f"P__{surf}__{cnd}__as0"] = pr[f"P__{src}__NL__as0"]
                        pr[f"V__{surf}__{cnd}__as0"] = pr[f"V__{src}__NL__as0"]
                        cand[cnd] = cnd
                lls = {c: log_loss(y[v_rows], pr[f"V__{surf}__{c}__as0"], labels=list(range(K))) for c in cand}
                best = min(lls, key=lls.get)
                sel[surf] = best
                for ak in AS:
                    if best in ("ignore_rep", "ignore_out"):
                        src = "outputs" if best == "ignore_rep" else "rep"
                        pr[f"P__{surf}__NL__as{ak}"] = pr[f"P__{src}__NL__as{ak}"]
                    else:
                        pr[f"P__{surf}__NL__as{ak}"] = pr[f"P__{surf}__{best}__as{ak}"]
                pr[f"V__{surf}__NL__as0"] = pr[f"V__{surf}__{best}__as0"]
            # held-out G1 / G2 / rho
            Y = onehot(y, range(K))
            g1 = Ridge(alpha=1e-6).fit(Hm[fm], Y[fm])
            pr["G1_pred"] = g1.predict(Hm[a_rows])
            pr["G1_prior"] = Y[fm].mean(0)
            mu, sd = Hm[fm].mean(0), Hm[fm].std(0)
            g2 = Ridge(alpha=1.0).fit((Hm[fm] - mu) / sd, Y[fm])
            pr["G2_pred"] = g2.predict((Hm[a_rows] - mu) / sd)
            pr["G2_prior"] = Y[fm].mean(0)
            cca = CCA(n_components=1).fit(Hm[fm], Y[fm][:, :-1])
            u_, v_ = cca.transform(Hm[a_rows], Y[a_rows][:, :-1])
            pr["RHO_u"], pr["RHO_v"] = u_.ravel(), v_.ravel()
            # label-only (Laplace 1) and utility
            lo = np.empty((len(a_rows), K))
            for t in (0, 1):
                ss = fm & (yt == t)
                lo[yt[a_rows] == t] = [(np.sum(y[ss] == c) + 1.0) / (ss.sum() + K) for c in range(K)]
            pr["LO_P"] = lo
            pr["U1_logits"] = G["logits"][s][a_rows] if arm_key == "A" else Hm[a_rows] @ whead
            if arm_key == "A":
                U2_A = LogisticRegression(C=1.0, max_iter=2000).fit(H[fm], yt[fm]).predict_proba(H[a_rows])
                pr["U2_P"] = U2_A
            else:
                f = U2_SWAP.get((name, grp), U2_SWAP.get((name, "B")) if arm_key == "C" else 0.0)
                U2 = U2_A.copy()
                sw = srng(name, s, arm_key, "u2").random(len(a_rows)) < f
                U2[sw] = U2[sw][:, ::-1]
                pr["U2_P"] = U2
            np.savez(udir / "preds.npz", **pr)
            (udir / "COMPLETE.json").write_text(json.dumps({"files": {
                f: sha(udir / f) for f in ("preds.npz", "supported.json", "fit_records.json")}}))
            truth.setdefault("selected", {})[uid] = sel
    return truth


def generate(work):
    if work.exists():
        shutil.rmtree(work)
    (work / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(20261002)
    index, G, truth = {"datasets": {}}, {}, {}
    for name in ("adult", "hmda"):
        index["datasets"][name], G[name] = gen_dataset(name, work, rng)
    (work / "inputs" / "INPUTS_INDEX.json").write_text(json.dumps(index, indent=1))
    for name in ("adult", "hmda"):
        truth[name] = gen_units(name, work, G[name])
    return G, truth


# ================================================================ reference implementation (independent)
def ref_auc(score, pos):
    """Rank-sum AUC with average ranks (independent of replay's weighted reduceat implementation)."""
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ref_unit_points(pr, y, yt, sup, prior):
    out = {}
    K = len(prior)
    for k in pr:
        if not (k.startswith("P__") or k == "LO_P"):
            continue
        P = np.asarray(pr[k], np.float64)
        if k == "LO_P":
            tag = "LO"
        else:
            _, surf, rest = k.split("__", 2)
            tag = "P|" + surf + "|" + rest.replace("__", "|")
        if len(sup) >= 2:
            cl = [roc_auc_score(y == c, P[:, c]) for c in sup]
            out[f"{tag}|AUC_macro"] = float(np.mean(cl))
            out[f"{tag}|AUC_worst_class"] = float(np.max(cl))
            prs = []
            for i, j in enumerate(sup):
                for kk in sup[i + 1:]:
                    m = (y == j) | (y == kk)
                    den = P[m, j] + P[m, kk]
                    prs.append(roc_auc_score(y[m] == kk, P[m, kk] / den))
            out[f"{tag}|AUC_worst_pair"] = float(np.max(prs))
            for c, v in zip(sup, cl):
                out[f"{tag}|AUC_class_{c}"] = float(v)
        pt = np.clip(P[np.arange(len(y)), y], 1e-12, 1 - 1e-12)
        pp = np.clip(prior[y], 1e-12, 1 - 1e-12)
        out[f"{tag}|logloss"] = float(-np.mean(np.log(pt)))
        out[f"{tag}|LL_skill"] = float(1 - np.mean(-np.log(pt)) / np.mean(-np.log(pp)))
        out[f"{tag}|LLR_nats"] = float(np.mean(np.log(pt) - np.log(pp)))
        Y = np.eye(K)[y]
        out[f"{tag}|brier_skill"] = float(1 - np.mean(((P - Y) ** 2).sum(1)) / np.mean(((prior - Y) ** 2).sum(1)))
    for g in ("G1", "G2"):
        Y = np.eye(K)[y]
        out[f"{g}|R2"] = float(1 - np.sum((Y - pr[f"{g}_pred"]) ** 2) / np.sum((Y - pr[f"{g}_prior"]) ** 2))
    out["RHO1SQ_heldout"] = float(np.corrcoef(pr["RHO_u"], pr["RHO_v"])[0, 1] ** 2)
    out["U1|accuracy"] = float(np.mean(np.argmax(pr["U1_logits"], 1) == yt))
    out["U2|accuracy"] = float(np.mean(np.argmax(pr["U2_P"], 1) == yt))
    return out


class RefModel:
    """Reference endpoints: members, statistic on a resampled row index, point estimates."""

    def __init__(self, work, G, sigma_star):
        self.work, self.G, self.sigma_star = work, G, sigma_star
        self.units = {}
        for p in sorted((work / "units").iterdir()):
            pm = RB.parse_unit_id(p.name)
            al = p / "ALIAS.json"
            src = json.loads(al.read_text())["alias_of"] if al.exists() else p.name
            self.units[p.name] = (pm, dict(np.load(work / "units" / src / "preds.npz")))

    def members(self, ds, grp):
        return [(u, pr) for u, (pm, pr) in self.units.items() if pm["ds"] == ds and pm["group"] == grp]

    def endpoint_fn(self, ds, fam, arm):
        cell = CELLS[ds]
        G = self.G[ds]
        g = f"D_sigma{fs(self.sigma_star[ds])}" if arm == "D" else arm
        role = G["role"]
        y_full = G["labels"][cell["target"]]
        mem = self.members(ds, g)
        first = mem[0][1]
        pos = {int(r): i for i, r in enumerate(G["row_id"])}
        idx = np.array([pos[int(r)] for r in first["assess_row_id"]])
        y = y_full[idx]
        yt = G["labels"][cell["task"]][idx]
        sup = own_support(y_full, role)
        if fam == "G1":
            K = int(y_full.max()) + 1
            Y = np.eye(K)[y]
            parts = [(pr["G1_pred"], pr["G1_prior"]) for _, pr in mem]

            def f(rows):
                return np.mean([1 - np.sum((Y[rows] - p[rows]) ** 2) / np.sum((Y[rows] - q) ** 2) for p, q in parts])
            return f, idx
        if fam in ("Rrep", "Rplus"):
            surf = "rep" if fam == "Rrep" else "repPLUSoutputs"
            Ps = [pr[f"P__{surf}__NL__as{a}"] for _, pr in mem for a in AS]

            def f(rows):
                return np.mean([np.mean([ref_auc(P[rows, c], y[rows] == c) for c in sup]) for P in Ps])
            return f, idx
        if fam == "U2NI":
            accs_m = [(np.argmax(pr["U2_P"], 1) == yt).astype(float) for _, pr in mem]
            accs_a = [(np.argmax(pr["U2_P"], 1) == yt).astype(float) for _, pr in self.members(ds, "A")]

            def f(rows):
                return np.mean([a[rows].mean() for a in accs_m]) - np.mean([a[rows].mean() for a in accs_a])
            return f, idx
        raise ValueError(fam)

    def worst_class_fns(self, ds, grp, surf):
        cell = CELLS[ds]
        G = self.G[ds]
        mem = self.members(ds, grp)
        pos = {int(r): i for i, r in enumerate(G["row_id"])}
        idx = np.array([pos[int(r)] for r in mem[0][1]["assess_row_id"]])
        y = G["labels"][cell["target"]][idx]
        sup = own_support(G["labels"][cell["target"]], G["role"])
        Ps = [pr[f"P__{surf}__NL__as{a}"] for _, pr in mem for a in AS]
        return {c: (lambda rows, c=c: np.mean([ref_auc(P[rows, c], y[rows] == c) for P in Ps])) for c in sup}, idx


def ref_bootstrap(fns, idx, units_of_rows, B, seed):
    """Index-resampling cluster bootstrap (units drawn with numpy RandomState, rows expanded)."""
    rs = np.random.RandomState(seed)
    uniq, inv = np.unique(units_of_rows, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    out = {k: np.empty(B) for k in fns}
    for b in range(B):
        draw = rs.randint(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_of[i] for i in draw])
        for k, f in fns.items():
            out[k][b] = f(rows)
    return out


def ref_sigma_star(work, G):
    out = {}
    for ds in ("adult", "hmda"):
        cell = CELLS[ds]
        pos = {int(r): i for i, r in enumerate(G[ds]["row_id"])}
        curve = {}
        for sg in SIGMAS:
            vals = []
            for s in ENC:
                for k in RS:
                    pr = np.load(work / "units" / f"{ds}__s{s}__{cell['purpose']}__{cell['target']}__D_sigma{fs(sg)}_rs{k}"
                                 / "preds.npz")
                    yv = G[ds]["labels"][cell["target"]][[pos[int(r)] for r in pr["val_row_id"]]]
                    sup = own_support(G[ds]["labels"][cell["target"]], G[ds]["role"])
                    vals.append(np.mean([roc_auc_score(yv == c, pr["V__rep__NL__as0"][:, c]) for c in sup]))
            curve[sg] = float(np.mean(vals))
        q = [sg for sg in sorted(SIGMAS) if curve[sg] <= 0.55]
        out[ds] = {"sigma_star": q[0] if q else max(SIGMAS), "flagged": not q, "curve": curve}
    return out


# ================================================================ checks
class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail=""):
        self.items.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        return ok


def unit_tests(C):
    rng = np.random.default_rng(5)
    # weighted AUC with ties vs sklearn sample_weight
    s = np.round(rng.normal(size=600), 1)
    yb = rng.random(600) < 0.4
    w = rng.integers(0, 4, size=600).astype(float)
    a = RB.WAUC(s, yb)(w[None, :])[0]
    b = roc_auc_score(yb, s, sample_weight=w)
    C.add("unit: weighted AUC with ties == sklearn roc_auc_score(sample_weight)", abs(a - b) < 1e-12, f"{a:.12f} vs {b:.12f}")
    # fixed ridge == sklearn Ridge(alpha=1e-6)
    H = rng.normal(size=(500, 8))
    Y = np.eye(3)[rng.integers(0, 3, 500)]
    p1, _ = RB.ridge_onehot(H[:300], Y[:300], H[300:])
    p2 = Ridge(alpha=1e-6).fit(H[:300], Y[:300]).predict(H[300:])
    C.add("unit: fixed ridge 1e-6 == sklearn Ridge(alpha=1e-6)", np.max(np.abs(p1 - p2)) < 1e-9,
          f"{np.max(np.abs(p1 - p2)):.2e}")
    # LEACE numpy re-derivation vs the official torch fitter: generic, truncated, trace-mixed problems
    probs = {}
    X = rng.normal(size=(2000, 12)) @ rng.normal(size=(12, 12))
    z = rng.integers(0, 4, 2000)
    X[:, 0] += 2 * (z == 1)
    probs["generic 4-class"] = (X, np.eye(4)[z])
    X2 = rng.normal(size=(2000, 12))
    z2 = rng.integers(0, 3, 2000)
    X2[:, 0] += 1.5 * (z2 == 1) + 0.002 * (z2 == 2)
    probs["weak class (svd_tol truncation)"] = (X2, np.eye(3)[z2])
    X3 = rng.normal(size=(2000, 6)) * np.array([10, 5, 1, 0.1, 0.05, 0.01])
    z3 = rng.integers(0, 2, 2000)
    X3[:, 5] += 0.3 * z3
    X3[:, 0] += 3 * z3
    probs["anisotropic (trace constraint)"] = (X3, np.eye(2)[z3])
    for nm, (Xp, Zp) in probs.items():
        Lm, Rm, b, er = fit_leace(Xp, Zp)
        nd = RB.leace_numpy(Xp, Zp)
        dP = np.max(np.abs((np.eye(Xp.shape[1]) - Lm @ Rm) - nd["P"]))
        Xe = RB.leace_apply(Xp, Lm, Rm, b)
        dX = np.max(np.abs(Xe - er(torch.from_numpy(Xp)).numpy()))
        xc = np.max(np.abs(RB.cross_cov(Xe, Zp)))
        C.add(f"unit: LEACE numpy re-derivation == official ({nm})", dP < RB.LEACE_P_TOL and dX < 1e-10,
              f"max|dP|={dP:.2e} max|dX|={dX:.2e} truncated={nd['n_truncated']} trace_mixed={nd['trace_mixed']} "
              f"residual xcov={xc:.2e}")
    C.add("unit: truncation case exercised (official eraser leaves residual cross-covariance)",
          RB.leace_numpy(*probs["weak class (svd_tol truncation)"])["n_truncated"] > 0)
    # bootstrap: duplicates collapse to one cluster
    sc = rng.normal(size=400) + (yy := rng.random(400) < 0.5)
    units = np.r_[np.arange(400), np.arange(400)]
    W1 = RB.run_bootstrap({"a": lambda c: RB.WAUC(np.r_[sc, sc], np.r_[yy, yy])(c.W)}, np.r_[np.arange(400), np.arange(400)],
                          400, 2000, 1)["a"]
    W2 = RB.run_bootstrap({"a": lambda c: RB.WAUC(sc, yy)(c.W)}, np.arange(400), 400, 2000, 1)["a"]
    C.add("unit: cluster bootstrap of duplicated rows == bootstrap of unique units", np.max(np.abs(W1 - W2)) < 1e-12,
          f"{np.max(np.abs(W1 - W2)):.1e}")
    # derived max bound is the max of per-component a/K quantiles
    comps = {0: rng.normal(0.6, 0.01, 4000), 1: rng.normal(0.6, 0.01, 4000), 2: rng.normal(0.5, 0.01, 4000)}
    d = RB.derived_max_bounds(comps, 0.05)
    want_lo = max(np.quantile(v, 0.05 / 3) for v in comps.values())
    want_hi = max(np.quantile(v, 1 - 0.05 / 3) for v in comps.values())
    C.add("unit: derived worst bound = max_k q_k(a/K), max_k q_k(1-a/K)", abs(d["lower"] - want_lo) < 1e-15 and
          abs(d["upper"] - want_hi) < 1e-15)
    # decisions incl. non-inferiority margin edge cases
    C.add("unit: decide_ni(LCB=-0.01) == NONINFERIOR (>= margin)", RB.decide_ni(-0.01, 0.02) == "NONINFERIOR")
    C.add("unit: decide_ni(UCB=-0.0101) == INFERIOR", RB.decide_ni(-0.05, -0.0101) == "INFERIOR")
    C.add("unit: decide_ni(UCB=-0.01) == UNRESOLVED (strict <)", RB.decide_ni(-0.05, -0.01) == "UNRESOLVED")
    C.add("unit: decide_bar strict both sides", RB.decide_bar(0.55, 0.6, 0.55) == "UNRESOLVED" and
          RB.decide_bar(0.5, 0.55, 0.55) == "UNRESOLVED" and RB.decide_bar(0.5501, 0.6, 0.55) == "ABOVE")
    # role rule vs separately typed copy
    keys = [rng.bytes(10).hex() for _ in range(3000)]
    C.add("unit: role rule (both salts) == separately typed copy",
          all(RB.role_of_key(SALTS[d_], k) == own_role(SALTS[d_], k) for d_ in SALTS for k in keys))


def null_calibration(C, n_rep=400, n=800, B=1000, K=4, n_members=3):
    """Coverage of the derived worst-class bound (exploratory 90 %: a_side = 0.05) under a null and a near-tied truth.
    Columns are independent per-class scores so the population class AUC is exact: Phi(a / sqrt(2))."""
    res = {}
    for label, a in (("null (all class AUC 0.5)", np.zeros(K)),
                     ("near-tied (0.60, 0.60, 0.59, 0.50)", np.sqrt(2) * norm.ppf([0.60, 0.60, 0.59, 0.50]))):
        theta = float(np.max(norm.cdf(a / np.sqrt(2))))
        rng = np.random.default_rng(99)
        cov_lo = cov_hi = cov_two = cov_naive_lo = cov_naive_hi = 0
        for r in range(n_rep):
            y = rng.choice(K, size=n, p=[0.4, 0.3, 0.2, 0.1])
            Ps = []
            for _ in range(n_members):
                Z = rng.normal(size=(n, K))
                Z[np.arange(n), y] += a[y]
                Ps.append(Z)
            stats = {}
            for c in range(K):
                ws = [RB.WAUC(P[:, c], y == c) for P in Ps]
                stats[c] = (lambda ws=ws: lambda ctx: np.mean([w(ctx.W) for w in ws], axis=0))()
            stats["max"] = (lambda st=dict(stats): lambda ctx: np.max([st[c](ctx) for c in range(K)], axis=0))()
            bs = RB.run_bootstrap(stats, np.arange(n), n, B, [r, 7])
            d = RB.derived_max_bounds({c: bs[c] for c in range(K)}, 0.05)
            cov_lo += d["lower"] <= theta
            cov_hi += d["upper"] >= theta
            cov_two += d["lower"] <= theta <= d["upper"]
            cov_naive_lo += np.quantile(bs["max"], 0.05) <= theta
            cov_naive_hi += np.quantile(bs["max"], 0.95) >= theta
        res[label] = {"theta_max": theta, "n_replicates": n_rep, "B": B, "n": n, "K": K, "members": n_members,
                      "coverage_LCB_one_sided": cov_lo / n_rep, "coverage_UCB_one_sided": cov_hi / n_rep,
                      "coverage_two_sided_90": cov_two / n_rep, "nominal_one_sided": 0.95,
                      "naive_bound_of_max_LCB_coverage": cov_naive_lo / n_rep,
                      "naive_bound_of_max_UCB_coverage": cov_naive_hi / n_rep}
        se = np.sqrt(0.95 * 0.05 / n_rep)
        C.add(f"null-calibration {label}: derived LCB(max) one-sided coverage >= 0.95 - 2 MC SE",
              cov_lo / n_rep >= 0.95 - 2 * se, json.dumps(res[label]))
        C.add(f"null-calibration {label}: derived UCB(max) one-sided coverage >= 0.95 - 2 MC SE",
              cov_hi / n_rep >= 0.95 - 2 * se)
    return res


def write_runner_report(rdir, prim_ref, ss_ref, worst_ref, B, seed):
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "point", "lower", "upper", "decision", "alpha_each", "B", "seed", "alias_of"])
        for r in prim_ref:
            w.writerow([r["id"], repr(r["point"]), repr(r["lower"]), repr(r["upper"]), r["decision"],
                        repr(0.05 / 24), B, seed, r.get("alias_of") or ""])
    (rdir / "SIGMA_STAR.json").write_text(json.dumps({d: {"sigma_star": v["sigma_star"], "flagged": v["flagged"]}
                                                      for d, v in ss_ref.items()}))
    with open(rdir / "WORST_CLASS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "point", "lower", "upper"])
        for k, v in worst_ref.items():
            w.writerow([k, repr(float(v["point"])), repr(float(v["lower"])), repr(float(v["upper"]))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(DEFAULT_WORK))
    ap.add_argument("--b-prim", type=int, default=20000)
    ap.add_argument("--b-expl", type=int, default=2000)
    ap.add_argument("--b-ref", type=int, default=20000)
    ap.add_argument("--null-reps", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "test_replay_bench_synthetic_result.json"))
    a = ap.parse_args(argv)
    work = Path(a.work)
    C = Checks()
    t0 = time.time()
    unit_tests(C)
    G, truth = generate(work / "run")
    root = work / "run"
    C.add("generator: hmda C map equals B map (alias case present)", all(truth["hmda"]["alias"].values()),
          truth["hmda"]["alias"])
    C.add("generator: adult C map differs from B map (non-alias case present)", not any(truth["adult"]["alias"].values()))
    cfg = RB.Config(sigma_grid=SIGMAS, encoder_seeds=ENC, release_seeds=RS, attacker_seeds=AS,
                    b_prim=a.b_prim, b_expl=a.b_expl, expected_counts=None)
    # ---------------- reference ground truth
    ss_ref = ref_sigma_star(root, G)
    C.add("design: adult sigma* = 0.5 (qualifies), hmda sigma* = max grid flagged",
          ss_ref["adult"]["sigma_star"] == 0.5 and not ss_ref["adult"]["flagged"] and
          ss_ref["hmda"]["sigma_star"] == 2.0 and ss_ref["hmda"]["flagged"], ss_ref)
    ref = RefModel(root, G, {d: v["sigma_star"] for d, v in ss_ref.items()})
    prim_ref = []
    t1 = time.time()
    for di, ds in enumerate(("adult", "hmda")):
        fns, idx = {}, None
        rows = [("G1", "A")] + [(f, m) for f in ("Rrep", "Rplus") for m in ("A", "B", "C", "D")] + \
               [("U2NI", m) for m in ("B", "C", "D")]
        for fam, arm in rows:
            fn, idx = ref.endpoint_fn(ds, fam, arm)
            fns[f"{fam}[{ds},{arm}]"] = fn
        units_rows = G[ds]["unit"][idx]
        allrows = np.arange(len(idx))
        bs = ref_bootstrap(fns, idx, units_rows, a.b_ref, 12345 + di)
        for fam, arm in rows:
            k = f"{fam}[{ds},{arm}]"
            x = bs[k]
            lo, hi = np.quantile(x, 0.05 / 24, method="linear"), np.quantile(x, 1 - 0.05 / 24, method="linear")
            dec = (RB.decide_ni(lo, hi) if fam == "U2NI" else RB.decide_bar(lo, hi, 0.05 if fam == "G1" else 0.55))
            alias = f"{fam}[{ds},B]" if (arm == "C" and all(truth[ds]["alias"].values())) else None
            prim_ref.append({"id": k, "point": float(fns[k](allrows)), "lower": float(lo), "upper": float(hi),
                             "decision": dec, "alias_of": alias, "lower_alpha05": float(np.quantile(x, 0.05)),
                             "mcse_lower": RB.mc_se_quantile(x, 0.05 / 24), "mcse_upper": RB.mc_se_quantile(x, 1 - 0.05 / 24)})
    print(f"[ref] primary reference bootstrap B={a.b_ref} done in {time.time() - t1:.0f}s", flush=True)
    worst_ref = {}
    for ds, grp, surf in (("hmda", "A", "rep"), ("adult", "A", "rep")):
        fns, idx = ref.worst_class_fns(ds, grp, surf)
        bs = ref_bootstrap(fns, idx, G[ds]["unit"][idx], a.b_expl, 777)
        K = len(fns)
        allrows = np.arange(len(idx))
        key = f"SUM|{ds}|{grp}|P|{surf}|NL|AUC_worst_class_derived"
        worst_ref[key] = {"point": max(f(allrows) for f in fns.values()),
                          "lower": max(np.quantile(v, 0.05 / K) for v in bs.values()),
                          "upper": max(np.quantile(v, 1 - 0.05 / K) for v in bs.values()),
                          "naive_lower": None, "per_class": {c: [float(np.quantile(v, 0.05 / K)),
                                                                 float(np.quantile(v, 1 - 0.05 / K))]
                                                             for c, v in bs.items()}}
        mx = np.max(np.stack(list(bs.values())), axis=0)
        worst_ref[key]["naive_lower"] = float(np.quantile(mx, 0.05))
        worst_ref[key]["naive_upper"] = float(np.quantile(mx, 0.95))
    rdir = work / "runner_report"
    write_runner_report(rdir, prim_ref, ss_ref, worst_ref, a.b_ref, 20261004)
    # ---------------- clean replay
    t2 = time.time()
    payload, res = RB.replay(root, rdir, work / "replay_clean.json", work / "replay_clean_results.json", cfg)
    t_clean = time.time() - t2
    summ = payload["summary"]
    bad = [i for i in payload["items"] if i["status"] in ("FAIL",)]
    C.add("clean replay: no FAIL items", not bad, f"summary={summ}; first fails={[b['check'] for b in bad[:8]]}")
    nonpass = [i for i in payload["items"] if i["status"] not in ("PASS", "INFO")]
    expected_np = [i for i in nonpass if i["status"] == "MC_BORDERLINE"]
    C.add("clean replay: every non-PASS/INFO item is MC_BORDERLINE (diagnosed)", len(nonpass) == len(expected_np),
          [(i["check"], i["status"]) for i in nonpass][:10])
    # ids / roles / support
    for ds in ("adult", "hmda"):
        g = G[ds]
        exp_df = (g["split"] == "train") & ~np.isin(g["ckey"], g["ckey"][g["split"] == "test"])
        C.add(f"{ds}: defense_fit excludes the {SPEC[ds]['n_shared']} shared-canon-key train rows (excluded_dup)",
              int((g["split"] == "train").sum() - exp_df.sum()) == SPEC[ds]["n_shared"] and
              int(np.sum(g["role"] == "excluded_dup")) == SPEC[ds]["n_shared"])
    sup = res["support"]
    C.add("support: hmda race classes 3 and 4 NE in every arm (macro over {0,1,2}, 3 pairs)",
          all(sup[u]["supported_classes"] == [0, 1, 2] for u in sup if u.startswith("hmda")),
          {u: sup[u]["supported_classes"] for u in sup if u.startswith("hmda__s0")})
    C.add("support: adult sex both classes supported in every arm",
          all(sup[u]["supported_classes"] == [0, 1] for u in sup if u.startswith("adult")))
    # every unit-level point estimate vs the reference
    maxd, n_cmp, worst_key = 0.0, 0, None
    pu = res["point_units"]
    for uid, (pm, pr) in ref.units.items():
        ds = pm["ds"]
        cell = CELLS[ds]
        pos = {int(r): i for i, r in enumerate(G[ds]["row_id"])}
        idx = np.array([pos[int(r)] for r in pr["assess_row_id"]])
        y = G[ds]["labels"][cell["target"]][idx]
        yt = G[ds]["labels"][cell["task"]][idx]
        fmask = G[ds]["role"] == "attacker_fit"
        yf = G[ds]["labels"][cell["target"]]
        prior = np.array([np.mean(yf[fmask] == c) for c in range(int(yf.max()) + 1)])
        sp = own_support(yf, G[ds]["role"])
        for k, v in ref_unit_points(pr, y, yt, sp, prior).items():
            rk = f"{uid}|{k}"
            if rk not in pu:
                C.add(f"point: replay computes {rk}", False)
                continue
            d_ = abs(pu[rk] - v)
            n_cmp += 1
            if d_ > maxd:
                maxd, worst_key = d_, rk
    C.add(f"point estimates: all {n_cmp} unit-level statistics (macro/worst class/pair AUC, Brier, LL, LLR, G1, G2, "
          f"rho1^2, U1, U2) == reference", maxd <= RB.POINT_TOL, f"max |diff| = {maxd:.2e} at {worst_key}")
    # replicate summaries vs reference means
    rp = {r["id"]: r for r in res["primary"]}
    maxd = max(abs(rp[r["id"]]["estimate"] - r["point"]) for r in prim_ref if "estimate" in rp[r["id"]])
    C.add("replicate summaries: 24 primary point estimates (mean over encoder x release x attacker seeds) == reference",
          maxd <= RB.POINT_TOL and len(rp) == 24, f"max |diff| = {maxd:.2e}")
    # primary bounds vs the reference bootstrap (independent RNG / algorithm)
    worst_ratio, bdet = 0.0, None
    for r in prim_ref:
        m = rp[r["id"]]
        for side in ("lower", "upper"):
            tol = 4.6 * np.hypot(m[f"mcse_{side}"], r[f"mcse_{side}"]) + RB.BOUND_FLOOR
            ratio = abs(m[side] - r[side]) / tol
            if ratio > worst_ratio:
                worst_ratio, bdet = ratio, f"{r['id']} {side}: replay {m[side]:.6f} ref {r[side]:.6f} tol {tol:.2e}"
    C.add("primary bounds (alpha=0.05/24, B=20000) agree with the reference bootstrap within MC tolerance",
          worst_ratio <= 1.0, f"worst |diff|/tol = {worst_ratio:.2f} ({bdet})")
    dec_ok = all(rp[r["id"]]["decision"] == r["decision"] or rp[r["id"]]["mc_borderline"] for r in prim_ref)
    C.add("primary decisions == reference decisions", dec_ok,
          {r["id"]: (rp[r["id"]]["decision"], r["decision"]) for r in prim_ref if rp[r["id"]]["decision"] != r["decision"]})
    designed = {"Rrep[adult,A]": "ABOVE", "Rrep[adult,B]": "BELOW", "Rrep[adult,C]": "UNRESOLVED",
                "U2NI[adult,B]": "NONINFERIOR", "U2NI[adult,C]": "INFERIOR", "U2NI[adult,D]": "UNRESOLVED",
                "Rrep[hmda,B]": "BELOW", "Rrep[hmda,C]": "BELOW", "U2NI[hmda,C]": "NONINFERIOR",
                "Rplus[adult,A]": "ABOVE"}
    C.add("designed decisions reproduced (null BELOW, borderline UNRESOLVED, NI margin -0.01 both ways)",
          all(rp[k]["decision"] == v for k, v in designed.items()),
          {k: rp[k]["decision"] for k in designed})
    C.add("family size 24 with hmda C endpoints as alias rows (counted, not dropped)",
          res["primary_family"]["size"] == 24 and all(rp[f"{f}[hmda,C]"]["alias_of"] == f"{f}[hmda,B]"
                                                      for f in ("Rrep", "Rplus", "U2NI")) and
          all(rp[f"{f}[hmda,C]"]["estimate"] == rp[f"{f}[hmda,B]"]["estimate"] for f in ("Rrep", "Rplus", "U2NI")))
    C.add("sigma* reproduced from attacker_val predictions only",
          all(res["sigma_star"][d]["sigma_star"] == ss_ref[d]["sigma_star"] and
              res["sigma_star"][d]["flagged_no_sigma_qualifies"] == ss_ref[d]["flagged"] for d in ss_ref),
          {d: res["sigma_star"][d]["sigma_star"] for d in ss_ref})
    # exploratory derived worst-class bounds vs reference (equal B = 2000)
    for key, w in worst_ref.items():
        m = res["worst_derived"][key]
        tl, th = RB.bound_tol(m["mcse_lower"]), RB.bound_tol(m["mcse_upper"])
        C.add(f"worst-class derived bound {key}: point (max of seed-mean class AUCs) and bounds == reference",
              abs(m["estimate_max_of_seedmeans"] - w["point"]) <= 1e-9 and abs(m["lower"] - w["lower"]) <= tl * 1.42
              and abs(m["upper"] - w["upper"]) <= th * 1.42,
              f"replay [{m['lower']:.4f}, {m['upper']:.4f}] ref [{w['lower']:.4f}, {w['upper']:.4f}] "
              f"naive bound-of-max [{w['naive_lower']:.4f}, {w['naive_upper']:.4f}]")
    hm = res["worst_derived"]["SUM|hmda|A|P|rep|NL|AUC_worst_class_derived"]
    pc = sorted(v[0] for v in hm["per_component"].values())
    C.add("near-tied worst class: top two per-class lower bounds within 0.01 (tie exercised)", pc[-1] - pc[-2] < 0.01,
          hm["per_component"])
    # exploratory 90 % intervals: a reference check on one summary at equal B
    k = "SUM|adult|A|P|rep|NL|AUC_macro"
    fn, idx = ref.endpoint_fn("adult", "Rrep", "A")
    bs = ref_bootstrap({"x": fn}, idx, G["adult"]["unit"][idx], a.b_expl, 4242)["x"]
    e = res["exploratory"][k]
    ok = abs(e["lower"] - np.quantile(bs, 0.05)) <= 1.42 * RB.bound_tol(e["mcse_lower"]) and \
        abs(e["upper"] - np.quantile(bs, 0.95)) <= 1.42 * RB.bound_tol(e["mcse_upper"])
    C.add("exploratory 90% interval (B=2000) == reference within MC tolerance", ok,
          f"replay [{e['lower']:.5f}, {e['upper']:.5f}] ref [{np.quantile(bs, 0.05):.5f}, {np.quantile(bs, 0.95):.5f}]")
    # LEACE properties
    lc = res["leace"]
    maps_ = {k_: v for k_, v in lc.items() if "alias" not in k_}
    C.add("LEACE: erased defense_fit rows == numpy re-derivation of the official fitter on every map",
          all(v["erased_fit_maxabs_diff_vs_rederivation"] <= 1e-6 * 10 for v in maps_.values()),
          {k_: f"{v['erased_fit_maxabs_diff_vs_rederivation']:.1e} (P {v['P_maxabs_diff_vs_rederivation']:.1e})"
           for k_, v in maps_.items()})
    C.add("LEACE native bound ||W P Sigma_xz||_2 <= svd_tol on every map (saved arrays and recomputed rows)",
          all(v["whitened_residual_spectral_saved"] <= 0.01 and v["whitened_residual_spectral_recomputed"] <= 0.01 + 1e-9
              for v in maps_.values()),
          {k_: (round(v["whitened_residual_spectral_saved"], 6), v["n_truncated_directions"]) for k_, v in maps_.items()})
    C.add("LEACE: exact-zero cross-covariance / OLS R^2 where nothing is truncated; truncation case exercised",
          all(v["xcov_after_relative"] <= RB.LEACE_XCOV_TOL and v["fit_ols_r2_after"] <= RB.LEACE_OLS_TOL
              for v in maps_.values() if v["n_truncated_directions"] == 0) and
          any(v["n_truncated_directions"] > 0 and v["xcov_after_relative"] > 1e-6 for v in maps_.values()),
          {k_: (f"{v['xcov_after_relative']:.1e}", v["n_truncated_directions"]) for k_, v in maps_.items()})
    C.add("LEACE: rank-deficient defense_fit covariance case present (hmda s0 dead unit)",
          lc["hmda__s0__underwriting__race__B"]["fit_rank_sample_cov"] < D)
    lci = [i for i in payload["items"] if i["scope"] == "leace" and i["status"] not in ("PASS", "INFO")]
    C.add("LEACE: every leace-scope item PASS/INFO in the clean run", not lci, [i["check"] for i in lci][:5])
    C.add("LEACE: B==C alias determined from saved maps (hmda alias, adult not)",
          all(lc[f"hmda__s{s}__alias"]["alias_replay"] for s in ENC) and
          not any(lc[f"adult__s{s}__alias"]["alias_replay"] for s in ENC))
    tr_items = [i for i in payload["items"] if "saved transformed rows" in i["check"]]
    C.add("LEACE: transformed attacker-role rows use the original fitting mean", tr_items and
          all(i["status"] == "PASS" for i in tr_items), len(tr_items))
    g1i = [i for i in payload["items"] if "G1_pred == fixed-ridge" in i["check"]]
    C.add("G1: saved G1_pred re-derived (A raw, B/C via saved map, D via pilot noise convention) on every unit",
          g1i and all(i["status"] == "PASS" for i in g1i), f"{len(g1i)} units")
    # ---------------- mutations (data): small B
    mcfg = RB.Config(sigma_grid=SIGMAS, encoder_seeds=ENC, release_seeds=RS, attacker_seeds=AS, b_prim=200, b_expl=200,
                     expected_counts=None)
    mutations = {}

    def mutate(name, fn, expect_rx):
        mdir = work / f"mut_{name}"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(root, mdir)
        ip = mdir / "inputs" / "INPUTS_INDEX.json"
        ip.write_text(ip.read_text().replace(str(root), str(mdir)))
        fn(mdir)
        pl, _ = RB.replay(mdir, None, None, None, mcfg, verbose=False)
        hits = [i["check"] for i in pl["items"] if i["status"] in ("FAIL", "NOTE") and __import__("re").search(expect_rx, i["check"])]
        mutations[name] = {"detected": bool(hits), "n_fail": pl["summary"].get("FAIL", 0), "matched": hits[:5]}
        C.add(f"mutation detected: {name}", bool(hits), hits[:3])
        shutil.rmtree(mdir)

    def m_contaminate(md):
        f = md / "inputs" / "adult_labels.npz"
        d = dict(np.load(f))
        i = np.flatnonzero(d["role"] == "assessment")[0]
        d["role"][i] = "defense_fit"
        np.savez(f, **d)
        _repin(md, "adult", "labels")
    mutate("defense_fit contaminated by an assessment row", m_contaminate,
           r"adult: (stored test-split roles|defense_fit canon keys disjoint|defense_fit units disjoint)")

    def m_support(md):
        u = md / "units" / "hmda__s0__underwriting__race__A" / "supported.json"
        j = json.loads(u.read_text())
        j["sensitive"]["supported_classes"] = [0, 1]
        u.write_text(json.dumps(j))
    mutate("supported.json drops a supported class", m_support, r"hmda__s0__underwriting__race__A: supported classes")

    def m_transform(md):
        dd = md / "defenses" / "adult__s1__income_prediction__sex__B"
        mp = dict(np.load(dd / "map.npz"))
        mp["mean"] = mp["mean_x"]
        t = dict(np.load(dd / "transformed.npz"))
        g = G["adult"]
        pos = {int(r): i for i, r in enumerate(g["row_id"])}
        idx = np.array([pos[int(r)] for r in t["row_id"]])
        Hh = g["H"][1][idx]
        out = np.empty_like(Hh)
        for r_ in np.unique(g["role"][idx]):
            sel = g["role"][idx] == r_
            out[sel] = RB.leace_apply(Hh[sel], mp["proj_left"], mp["proj_right"], Hh[sel].mean(0))
        t["H"] = out
        np.savez(dd / "transformed.npz", **t)
    mutate("transform uses each role's own mean", m_transform, r"adult__s1__income_prediction__sex__B: saved transformed")

    def m_alias(md):
        u = md / "units" / "adult__s0__income_prediction__sex__C"
        (u / "preds.npz").unlink()
        (u / "ALIAS.json").write_text(json.dumps({"alias_of": "adult__s0__income_prediction__sex__B"}))
    mutate("false B==C alias claim", m_alias, r"adult__s0: B==C alias claim")

    def m_g1(md):
        f = md / "units" / "hmda__s1__underwriting__race__B" / "preds.npz"
        d = dict(np.load(f))
        d["G1_pred"] = d["G1_pred"] * 1.01
        np.savez(f, **d)
    mutate("G1_pred not the fixed-ridge fit", m_g1, r"hmda__s1__underwriting__race__B: saved G1_pred")

    def m_salt(md):
        f = md / "inputs" / "hmda_labels.npz"
        d = dict(np.load(f))
        d["role"] = np.array([own_role("pilot-roles-v1|", k) if s_ == "test" else r
                              for k, s_, r in zip(d["record_key"], d["split"], d["role"])])
        np.savez(f, **d)
        _repin(md, "hmda", "labels")
    mutate("hmda roles hashed with the Adult salt", m_salt, r"hmda: stored test-split roles")

    def m_concept(md):
        dd = md / "defenses" / "adult__s0__income_prediction__sex__B"
        g = G["adult"]
        dfm = g["role"] == "defense_fit"
        save_map(dd, g["H"][0][dfm], onehot(g["labels"]["income"][dfm], [0, 1]))
    mutate("B map fitted on the wrong concept", m_concept,
           r"adult__s0__income_prediction__sex__B: (saved sigma_xz ==|NATIVE .*recomputed rows|erased defense_fit rows ==)")

    def m_mean(md):
        dd = md / "defenses" / "hmda__s0__underwriting__race__B"
        mp = dict(np.load(dd / "map.npz"))
        g = G["hmda"]
        mp["mean_x"] = g["H"][0][g["role"] == "attacker_fit"].mean(0)
        np.savez(dd / "map.npz", **mp)
    mutate("saved mean is not the defense_fit mean", m_mean, r"hmda__s0__underwriting__race__B: saved mean_x")

    def m_ignore(md):
        f = md / "units" / "adult__s1__income_prediction__sex__A" / "preds.npz"
        d = dict(np.load(f))
        d["P__repPLUSoutputs__ignore_rep__as0"] = d["P__repPLUSoutputs__GBT__as0"]
        np.savez(f, **d)
    mutate("ignore-rep candidate is not the outputs-only model", m_ignore,
           r"adult__s1__income_prediction__sex__A: repPLUSoutputs 'ignore_rep'")

    def m_outputs(md):
        f = md / "units" / "hmda__s1__underwriting__race__D_sigma2_rs0" / "preds.npz"
        d = dict(np.load(f))
        d["P__outputs__GBT__as1"] = d["P__outputs__GBT__as1"][::-1]
        np.savez(f, **d)
    mutate("outputs-only surface not aliased across methods", m_outputs, r"D_sigma2_rs0: outputs/GBT/as1 identical")

    # ---------------- mutations (runner report): reuse the clean replay results
    def report_mut(name, fn, rx):
        md = work / f"rep_{name}"
        if md.exists():
            shutil.rmtree(md)
        shutil.copytree(rdir, md)
        fn(md)
        rr = RB.Report()
        RB.compare_runner_reports(md, res, rr)
        hits = [i["check"] for i in rr.items if i["status"] == "FAIL" and __import__("re").search(rx, i["check"])]
        mutations[name] = {"detected": bool(hits), "matched": hits[:5]}
        C.add(f"mutation detected: {name}", bool(hits), hits[:3])
        shutil.rmtree(md)

    def edit_csv(path, fn):
        rows = list(csv.DictReader(open(path)))
        fn(rows)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    report_mut("runner point estimate perturbed by 1e-4",
               lambda md: edit_csv(md / "PRIMARY_ENDPOINTS.csv", lambda rs: rs[3].update(point=repr(float(rs[3]["point"]) + 1e-4))),
               r": point$")
    report_mut("runner decision flipped (Rrep[adult,A] ABOVE -> UNRESOLVED)",
               lambda md: edit_csv(md / "PRIMARY_ENDPOINTS.csv", lambda rs: [r.update(decision="UNRESOLVED")
                                                                            for r in rs if r["id"] == "Rrep[adult,A]"]),
               r"Rrep\[adult,A\]: decision$")
    report_mut("runner family has 23 endpoints", lambda md: edit_csv(md / "PRIMARY_ENDPOINTS.csv", lambda rs: rs.pop()),
               r"family size 24|family ids")
    l05 = {r["id"]: r["lower_alpha05"] for r in prim_ref}
    report_mut("runner one-sided bounds at alpha=0.05 instead of 0.05/24 (all rows)",
               lambda md: edit_csv(md / "PRIMARY_ENDPOINTS.csv", lambda rs: [r.update(lower=repr(l05[r["id"]]))
                                                                            for r in rs]),
               r"simultaneous lower bound")
    report_mut("runner sigma* wrong",
               lambda md: (md / "SIGMA_STAR.json").write_text(json.dumps({"adult": {"sigma_star": 2.0, "flagged": False},
                                                                         "hmda": {"sigma_star": 2.0, "flagged": True}})),
               r"adult: sigma\*$")
    report_mut("runner alias row missing for an aliased C endpoint",
               lambda md: edit_csv(md / "PRIMARY_ENDPOINTS.csv",
                                   lambda rs: [r.update(alias_of="") for r in rs if r["id"] == "Rrep[hmda,C]"]),
               r"Rrep\[hmda,C\]: alias_of")

    def wc_naive(md):
        def f(rs):
            for r in rs:
                w = worst_ref[r["key"]]
                r.update(lower=repr(float(w["naive_lower"])), upper=repr(float(w["naive_upper"])))
        edit_csv(md / "WORST_CLASS.csv", f)
    report_mut("runner worst-class bound = bound of the max (not max of per-class a/K bounds)", wc_naive,
               r"derived (lower|upper) bound")

    # ---------------- null calibration
    nc = null_calibration(C, n_rep=a.null_reps)
    out = {"schema": "pcrl.matched_removal_benchmark.replay_synthetic_validation/v1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "replay_bench_sha256": sha(HERE / "replay_bench.py"), "test_sha256": sha(Path(__file__)),
           "workdir": str(work), "B_primary": a.b_prim, "B_exploratory": a.b_expl, "B_reference": a.b_ref,
           "clean_replay_seconds": round(t_clean, 1), "clean_replay_summary": summ,
           "clean_replay_non_pass": [(i["check"], i["status"]) for i in nonpass],
           "tolerances_frozen_before_real_run": payload["tolerances"],
           "mutations": mutations, "null_calibration": nc,
           "n_checks": len(C.items), "n_pass": sum(i["pass"] for i in C.items), "checks": C.items,
           "elapsed_s": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(RB._jsonable(out), indent=1))
    print(f"\n{out['n_pass']}/{out['n_checks']} checks pass; {time.time() - t0:.0f}s", flush=True)
    return 0 if out["n_pass"] == out["n_checks"] else 1


def _repin(md, ds, what):
    ip = md / "inputs" / "INPUTS_INDEX.json"
    idx = json.loads(ip.read_text())
    e = idx["datasets"][ds]
    e[f"{what}_sha256"] = sha(e[f"{what}_npz"])
    ip.write_text(json.dumps(idx))


if __name__ == "__main__":
    sys.exit(main())
