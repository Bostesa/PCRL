"""Synthetic fixtures (the only data on which this package fits anything without --execute-scientific-fits).

Every fixture returns row-level arrays with explicit row ids, grouping units, roles and record keys, in
the same layout an admitted manifest produces, so the whole pipeline runs end to end on it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROLE_SHARES = (("attacker_fit", 0.5), ("attacker_val", 0.2), ("evaluation", 0.3))


def _roles_for_units(n_units, rng):
    u = rng.permutation(n_units)
    roles = np.empty(n_units, dtype=object)
    start = 0
    for i, (name, share) in enumerate(ROLE_SHARES):
        stop = n_units if i == len(ROLE_SHARES) - 1 else start + int(round(share * n_units))
        roles[u[start:stop]] = name
        start = stop
    return roles.astype(str)


def make_synthetic(kind: str = "direct", n_units: int = 2000, d: int = 8, seed: int = 0,
                   rows_per_unit: int = 1, dup_factor: int = 1, K: int = 2, signal: float = 1.0) -> dict:
    """kind: direct | null | xor | contrast | output_leak.

    direct      S is (noisily) a column of H                       (positive control)
    null        H independent of S                                 (null control)
    xor         S = 1[h0 * h1 > 0]: zero linear cross-covariance, recoverable by trees/MLP
    contrast    K classes; one direction encodes 1[y=a] - 1[y=b] for two minority classes only
    output_leak H carries nothing about S; the task outputs do (surface decomposition)
    dup_factor  every record is emitted dup_factor times with NEW row ids and NEW unit ids but the same
                record key (duplicated records must collapse to one unit in inference)
    """
    rng = np.random.default_rng(seed)
    n = n_units * rows_per_unit
    unit = np.repeat(np.arange(n_units), rows_per_unit)
    H = rng.normal(size=(n, d))
    if kind == "direct":
        S = rng.integers(0, 2, n)
        H[:, 0] = signal * (2 * S - 1) + 0.5 * rng.normal(size=n)
    elif kind == "null":
        S = rng.integers(0, 2, n)
    elif kind == "xor":
        S = (H[:, 0] * H[:, 1] > 0).astype(int)
    elif kind == "contrast":
        p = np.r_[0.03, 0.03, np.full(K - 2, 0.94 / (K - 2))]
        S = rng.choice(K, n, p=p)
        c = (S == 0).astype(float) - (S == 1).astype(float)
        H[:, 0] = signal * c + 0.05 * rng.normal(size=n)
    elif kind == "output_leak":
        S = rng.integers(0, 2, n)
    else:
        raise ValueError(kind)
    T = (H[:, 2] + 0.5 * rng.normal(size=n) > 0).astype(int)  # task label
    logit = 1.5 * H[:, 2]
    if kind == "output_leak":
        logit = logit + 1.2 * (2 * S - 1)
    outputs = np.c_[-logit / 2, logit / 2]
    roles = _roles_for_units(n_units, rng)[unit]
    rec = np.array([hashlib.sha256(H[i].tobytes() + bytes([int(S[i])])).hexdigest()[:16] for i in range(n)])
    if dup_factor > 1:
        H, S, T, outputs, roles, rec = (np.repeat(a, dup_factor, axis=0) for a in (H, S, T, outputs, roles, rec))
        unit = np.arange(len(S))  # duplicates get fresh unit ids: only the record key reveals them
    row_ids = rng.permutation(len(S)) + 10_000  # ids are NOT positions
    return {"H": H, "S": S.astype(np.int64), "T": T.astype(np.int64), "outputs": outputs, "units": unit,
            "roles": roles, "record_keys": rec, "row_ids": row_ids, "kind": kind, "synthetic": True}


def write_manifest(fx: dict, out_dir: str | Path, name: str = "synthetic") -> Path:
    """Write a fixture as an npz + manifest pair (for exercising `admit` and the CLI)."""
    from .admission import sha256_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{name}.npz"
    np.savez(npz, row_id=fx["row_ids"], H=fx["H"], S=fx["S"], T=fx["T"], outputs=fx["outputs"],
             unit=fx["units"], role=fx["roles"], record_key=fx["record_keys"])
    man = {"schema": "stored_model_eval.manifest/v1", "synthetic": True,
           "files": {"data": {"path": npz.name, "sha256": sha256_file(npz)}},
           "arrays": {"representations": {"file": "data", "key": "H", "ids": "row_id"},
                      "outputs": {"file": "data", "key": "outputs", "ids": "row_id"},
                      "labels": {"file": "data", "key": "S", "ids": "row_id"},
                      "task_labels": {"file": "data", "key": "T", "ids": "row_id"},
                      "units": {"file": "data", "key": "unit", "ids": "row_id"},
                      "roles": {"file": "data", "key": "role", "ids": "row_id"},
                      "record_keys": {"file": "data", "key": "record_key", "ids": "row_id"}},
           "min_class_support": 20}
    mp = out / f"{name}_manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    return mp


# --------------------------------------------------------------------------------------------------
# synthetic CELL-A world for the pilot runner (same file layout as ~/PCRL_eval_cache_private/pilot_adult_s0)
# --------------------------------------------------------------------------------------------------

PILOT_PAIRS = [("income_prediction", 0, "race"), ("income_prediction", 0, "sex"), ("employment_analysis", 1, "race"),
               ("employment_analysis", 1, "age_group"), ("employment_analysis", 1, "marital_status"),
               ("education_assessment", 2, "sex"), ("education_assessment", 2, "race"),
               ("education_assessment", 2, "income")]
HEADS = (("income_prediction", 2), ("employment_analysis", 6), ("education_assessment", 4))


def make_synthetic_checkpoint(path: Path, d: int, input_dim: int = 12, hidden: int = 16, rank: int = 2,
                              seed: int = 0) -> tuple[str, dict]:
    """A state dict with the PCRL v2 layout FrozenPCRLv2 accepts. Heads are exactly linear in the representation
    (W0 = [I; -I], relu(x) - relu(-x) = x), so head(rep) = rep @ A.T + b with a known A per purpose."""
    import torch
    g = torch.Generator().manual_seed(seed)
    bb = {}
    dims = [input_dim, hidden, hidden]
    for li, i in enumerate((0, 4)):
        bb[f"network.{i}.weight"] = torch.randn(dims[li + 1], dims[li], generator=g) * 0.1
        bb[f"network.{i}.bias"] = torch.zeros(dims[li + 1])
        bn = i + 1
        bb[f"network.{bn}.weight"] = torch.ones(dims[li + 1])
        bb[f"network.{bn}.bias"] = torch.zeros(dims[li + 1])
        bb[f"network.{bn}.running_mean"] = torch.zeros(dims[li + 1])
        bb[f"network.{bn}.running_var"] = torch.ones(dims[li + 1])
        bb[f"network.{bn}.num_batches_tracked"] = torch.tensor(1)
    bb["repr_proj.weight"] = torch.randn(d, hidden, generator=g) * 0.1
    bb["repr_proj.bias"] = torch.zeros(d)
    ad = {}
    ins, outs = [input_dim, hidden, hidden], [hidden, hidden, d]
    for p in range(3):
        for l in range(3):
            ad[f"{p}.{l}.A.weight"] = torch.zeros(rank, ins[l])
            ad[f"{p}.{l}.B.weight"] = torch.zeros(outs[l], rank)
            ad[f"{p}.{l}.bias"] = torch.zeros(outs[l])
    th, A = {}, {}
    for j, (name, C) in enumerate(HEADS):
        a = np.zeros((C, d), dtype=np.float32)
        col = 1 + j
        a[:, col] = np.linspace(-2.0, 2.0, C)            # ordinal classes along one representation column
        b = np.zeros(C, dtype=np.float32)
        if name == "employment_analysis":
            b[5] = -30.0                                   # class 5 essentially never predicted / drawn
        th[f"{name}.network.0.weight"] = torch.cat([torch.eye(d), -torch.eye(d)])
        th[f"{name}.network.0.bias"] = torch.zeros(2 * d)
        th[f"{name}.network.3.weight"] = torch.as_tensor(np.c_[a, -a])
        th[f"{name}.network.3.bias"] = torch.as_tensor(b)
        A[name] = (a, b)
    ck = {"backbone": bb, "lora_adapters": ad, "task_heads": th,
          "config": {"lora_rank": rank, "lora_alpha": float(2 * rank)}, "state": {"epoch": 0}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    from .admission import sha256_file
    return sha256_file(path), A


def make_pilot_world(root: str | Path, n: int = 2400, d: int = 8, seed: int = 0, sex_signal: float = 1.5) -> dict:
    """Write a synthetic world with the real pilot's layout: labels.npz, features.npz, cache/adult_s0_test.npz,
    releases/p{k}_sigma*_seed*.npz, the 152 preparation-style manifests, a synthetic checkpoint, and a task-label
    source (row ids permuted, so alignment must go through IDs).

    Signals: sex is directly encoded in rep_p0 column 0 (positive control); race and marital_status are independent
    of every representation (null controls; race classes 3 and 4 are below support); income is encoded in rep_p0
    column 1 (it is the income_prediction task); task labels are drawn from softmax(head(rep)) so U1 is meaningful.
    """
    from .admission import sha256_file
    from .releases import DOCUMENTED_ADULT_SIGMAS, gaussian_release
    root = Path(root)
    orig = root / "orig"
    (orig / "cache").mkdir(parents=True, exist_ok=True)
    (orig / "releases").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    ck_path = root / "checkpoints" / "synthetic.pt"
    ck_sha, A = make_synthetic_checkpoint(ck_path, d, seed=seed)
    sex = rng.integers(0, 2, n)
    race = rng.choice(5, n, p=[0.3, 0.3, 0.34, 0.03, 0.03])
    age = rng.choice(4, n, p=[0.25, 0.25, 0.25, 0.25])
    marital = rng.integers(0, 2, n)
    reps = []
    for p in range(3):
        H = rng.normal(size=(n, d))
        if p == 0:
            H[:, 0] = sex_signal * (2 * sex - 1) + 0.7 * rng.normal(size=n)
        if p == 1:
            H[:, 0] = 0.4 * (age - 1.5) + rng.normal(size=n)
        reps.append(H.astype(np.float32))

    def head(name, H):
        a, b = A[name]
        return (H.astype(np.float32) @ a.T + b).astype(np.float32)

    def draw(L):
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        return np.array([rng.choice(P.shape[1], p=q) for q in P])

    y_inc = draw(head("income_prediction", reps[0]).astype(np.float64))
    y_occ = draw(head("employment_analysis", reps[1]).astype(np.float64))
    y_edu = draw(head("education_assessment", reps[2]).astype(np.float64))
    income = y_inc
    rec = np.array([hashlib.sha256(f"{i}|{sex[i]}|{race[i]}|{age[i]}".encode()).hexdigest()[:20] for i in range(n)])
    rec[1::97] = rec[0:-1:97][: len(rec[1::97])]           # ~1% exact duplicate records -> shared unit
    _, unit = np.unique(rec, return_inverse=True)

    def role_of(key):
        u = int(hashlib.sha256(("pilot-roles-v1|" + key).encode()).hexdigest()[:8], 16) / 2 ** 32
        return "attacker_fit" if u < 0.5 else ("attacker_val" if u < 0.65 else "assessment")
    roles = np.array([role_of(k) for k in rec])
    row_id = np.arange(n, dtype=np.int64)
    np.savez(orig / "labels.npz", row_id=row_id, unit=unit, role=roles, record_key=rec, sex=sex, race=race,
             age_group=age, marital_status=marital, income=income)
    np.savez(orig / "features.npz", row_id=row_id, features=rng.normal(size=(n, 12)).astype(np.float32))
    fwd = orig / "cache" / "adult_s0_test.npz"
    np.savez(fwd, row_id=row_id, rep_p0=reps[0], rep_p1=reps[1], rep_p2=reps[2],
             **{f"logits_{name}": head(name, reps[j]) for j, (name, _) in enumerate(HEADS)})
    (orig / "forward_manifest.json").write_text(json.dumps({
        "schema": "stored_model_eval.manifest/v1", "synthetic": True, "required_arrays": ["features"],
        "files": {"checkpoint": {"path": str(ck_path), "sha256": ck_sha},
                  "features": {"path": "features.npz", "sha256": sha256_file(orig / "features.npz")}},
        "arrays": {"features": {"file": "features", "key": "features", "ids": "row_id"}}}, indent=1))
    fsha, lsha = sha256_file(fwd), sha256_file(orig / "labels.npz")

    def base_manifest(purpose, p, attr):
        return {"schema": "stored_model_eval.manifest/v1", "synthetic": True, "min_class_support": 100,
                "allowed_roles": ["defense_fit", "attacker_fit", "attacker_val", "assessment"],
                "files": {"fwd": {"path": str(fwd), "sha256": fsha},
                          "lab": {"path": str(orig / "labels.npz"), "sha256": lsha}},
                "arrays": {"representations": {"file": "fwd", "key": f"rep_p{p}", "ids": "row_id"},
                           "outputs": {"file": "fwd", "key": f"logits_{purpose}", "ids": "row_id"},
                           "labels": {"file": "lab", "key": attr, "ids": "row_id"},
                           "units": {"file": "lab", "key": "unit", "ids": "row_id"},
                           "roles": {"file": "lab", "key": "role", "ids": "row_id"},
                           "record_keys": {"file": "lab", "key": "record_key", "ids": "row_id"}}}
    n_man = 0
    for purpose, p, attr in PILOT_PAIRS:
        (orig / f"manifest_{purpose}__{attr}.json").write_text(json.dumps(base_manifest(purpose, p, attr), indent=1))
        n_man += 1
        for sigma in DOCUMENTED_ADULT_SIGMAS:
            for sd in (0, 1, 2):
                tag = f"p{p}_sigma{sigma:g}_seed{sd}"
                rf = orig / "releases" / f"{tag}.npz"
                if not rf.exists():
                    np.savez(rf, row_id=row_id, rep=gaussian_release(reps[p], sigma, sd).astype(np.float32))
                man = base_manifest(purpose, p, attr)
                man["release"] = {"kind": "gaussian_noise", "sigma_abs": sigma, "seed": sd, "release_count": "one",
                                  "outputs_surface": "clean model output"}
                man["files"]["rel"] = {"path": str(rf), "sha256": sha256_file(rf)}
                man["arrays"]["representations"] = {"file": "rel", "key": "rep", "ids": "row_id"}
                (orig / f"manifest_{purpose}__{attr}__{tag}.json").write_text(json.dumps(man, indent=1))
                n_man += 1
    perm = rng.permutation(n)
    source = {"row_id": row_id[perm], "y_task_income_prediction": y_inc[perm],
              "y_task_employment_analysis": y_occ[perm], "y_task_education_assessment": y_edu[perm],
              "ref_record_key": rec[perm], "ref_sex": sex[perm], "ref_race": race[perm], "ref_age_group": age[perm],
              "ref_marital_status": marital[perm], "ref_income": income[perm]}
    return {"root": root, "orig": orig, "n_manifests": n_man, "checkpoint": {"path": str(ck_path), "sha256": ck_sha},
            "source": source, "features": str(orig / "features.npz")}
