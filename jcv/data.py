"""Real-data admission for the joint complete-view study (Adult; development data already used by both repositories).

Raw rows are regenerated with the pinned loader (pcrl.data.adult.AdultDataset at b96c412; file unchanged since) and
matched to the admitted role manifest by record key (sha256 of the raw row, first 20 hex). Roles are oar-roles-v1
(exposure-cleaned, cert carve-out, head holdout = defense validation), reused exactly.

Corrected input contract (fixed before any fit):
  removed  sex, race                  protected (primary SEX; race = secondary stress audit)
           income                     task-1 label
           occupation                 task-2 label source (occupation_group is a many-to-one map of it)
           fnlwgt                     census record weight, not a person attribute
           row ids / record keys      never inputs
  kept     age, workclass, education, education-num, marital-status, relationship, capital-gain, capital-loss,
           hours-per-week, native-country
Preprocessing (standardisation of numeric columns) is fitted on the defense-training role only; categorical columns
use the loader's fixed category sets (no outcome-dependent choice).

    ~/PCRL/.venv/bin/python -m jcv.data      (writes the private inputs npz + public DATA_ADMISSION.json)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
PRIV = HOME / "PCRL_eval_cache_private" / "jcv_v1"
INPUTS = PRIV / "inputs" / "adult_jcv.npz"
PKG = WT / "results" / "pcrl_joint_complete_view_method_v1"
RAW_SHA = {"adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
           "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}
LABELS_SHA = "786a51f28ae567e7b32339514aeb320f1258c29fe574dde3fb65ba3574482832"

REMOVED = {"sex": "protected (primary)", "race": "protected (secondary audit)", "income": "task-1 label",
           "occupation": "task-2 label source (occupation_group = many-to-one map of occupation)",
           "fnlwgt": "census sampling weight (record weight), not a person attribute"}
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "relationship", "native-country"]
TASKS = {"income": 2, "occupation_group": 6}
PURPOSES = ("income", "occupation_group")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype.kind == "U":
        return hashlib.sha256("\n".join(a.tolist()).encode()).hexdigest()
    return hashlib.sha256(a.tobytes()).hexdigest()


def record_key(row) -> str:
    return hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:20]


def raw_frames():
    """Post-dropna raw rows of the pinned loader's test split (row_id 0..n_te-1) and train split (n_te + position)."""
    sys.path.insert(0, str(WT))
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    root = HOME / "PCRL" / "data"
    for f, h in RAW_SHA.items():
        if sha_file(root / "adult" / f) != h:
            raise SystemExit(f"REFUSED: raw file {f} does not match the admitted hash")
    P = get_adult_purposes()
    tr = AdultDataset(purposes=P, root=str(root), split="train", download=False)
    te = AdultDataset(purposes=P, root=str(root), split="test", download=False, norm_stats=tr.norm_stats)
    return te.raw_df.reset_index(drop=True), tr.raw_df.reset_index(drop=True)


def build():
    import oar.study as S
    te, tr = raw_frames()
    raw = pd.concat([te, tr], ignore_index=True)
    lab = np.load(S.BENCH / "inputs" / "adult_labels.npz")
    if sha_file(S.BENCH / "inputs" / "adult_labels.npz") != LABELS_SHA:
        raise SystemExit("REFUSED: admitted label manifest hash changed")
    rk = np.array([record_key(r) for r in raw.itertuples(index=False)])
    if not np.array_equal(rk, lab["record_key"].astype(str)):
        raise SystemExit("REFUSED: regenerated raw rows do not match the admitted record keys")
    W = S.load_world("adult")  # oar-roles-v1: exposure-cleaned, cert carve-out, head holdout
    assert np.array_equal(W["row_id"], np.arange(len(raw)))
    # labels from source definitions (pinned loader maps), re-derived and cross-checked
    from pcrl.data.adult import OCCUPATION_GROUPS
    sex = raw["sex"].map({"Female": 0, "Male": 1}).to_numpy().astype(np.int64)
    race_cats = sorted(raw["race"].unique().tolist())
    race = raw["race"].map({c: i for i, c in enumerate(race_cats)}).to_numpy().astype(np.int64)
    income = raw["income"].astype(str).str.rstrip(".").map({"<=50K": 0, ">50K": 1}).to_numpy().astype(np.int64)
    occ = raw["occupation"].map(OCCUPATION_GROUPS).to_numpy().astype(np.int64)
    for name, mine, theirs in (("sex", sex, lab["sex"]), ("race", race, lab["race"]), ("income", income, lab["task_income"]),
                               ("occupation_group", occ, lab["task_occupation_group"])):
        if not np.array_equal(mine, theirs):
            raise SystemExit(f"REFUSED: re-derived {name} differs from the admitted labels")
    # roles
    role = W["role"].astype("<U32").copy()
    role[W["head_val"]] = "defense_val"
    role[np.isin(role, ["defense_fit"])] = "defense_train"
    idx = {r: np.flatnonzero(role == r) for r in ("defense_train", "defense_val", "attacker_fit", "attacker_val",
                                                   "assessment", "cert", "excluded_exposure", "excluded_dup")}
    # permitted inputs; preprocessing fitted on defense_train only
    from pcrl.data.adult import AdultDataset
    cats = AdultDataset.CATEGORY_VALUES
    fit = idx["defense_train"]
    cols, names = [], []
    norm = {}
    for c in NUMERIC:
        v = raw[c].to_numpy().astype(np.float64)
        mu, sd = float(v[fit].mean()), float(v[fit].std())
        norm[c] = [mu, sd]
        cols.append(((v - mu) / sd)[:, None])
        names.append(c)
    for c in CATEGORICAL:
        oh = pd.get_dummies(pd.Categorical(raw[c], categories=cats[c])).to_numpy().astype(np.float64)
        cols.append(oh)
        names += [f"{c}={v}" for v in cats[c]]
    X = np.concatenate(cols, 1).astype(np.float32)
    out = {"row_id": W["row_id"], "unit": W["unit"], "role": role, "X": X, "feature_names": np.array(names),
           "sex": sex, "race": race, "y_income": income, "y_occupation_group": occ}
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INPUTS, **out)
    # shortcut audit (documentation, no fit)
    rel = raw["relationship"].to_numpy()
    husband_wife = np.isin(rel, ["Husband", "Wife"])
    sex_given_rel = {r: {"rows": int((rel == r).sum()), "share_male": float(sex[rel == r].mean())}
                     for r in cats["relationship"]}
    wc = raw["workclass"].to_numpy()
    occ_by_wc = {w: {"rows": int((wc == w).sum()),
                     "max_class_share": float(np.bincount(occ[wc == w], minlength=6).max() / max(1, (wc == w).sum()))}
                 for w in cats["workclass"] if (wc == w).sum()}
    adm = {
        "schema": "jcv-data-admission-v1", "population": "UCI Adult (development data already used in both repositories)",
        "raw_files_sha256": RAW_SHA, "label_manifest_sha256": LABELS_SHA,
        "loader": "pcrl.data.adult.AdultDataset (file unchanged since b96c41256daeed6e644aba1443a47b16e28d089a); post-dropna rows",
        "row_id_convention": "test rows 0..15059 (adult.test position), train rows 15060+ (80% seed-42 permutation of adult.data)",
        "record_key_match": "all 39,205 regenerated raw rows match the admitted record keys",
        "labels": {
            "income": "binary: '>50K' (trailing '.' of adult.test stripped) -> 1, '<=50K' -> 0; matches admitted task_income",
            "occupation_group": {"classes": 6, "map": OCCUPATION_GROUPS,
                                 "note": "6-class occupation GROUP (not unemployment); rows with missing occupation were dropped by the loader"},
            "sex": "Female 0 / Male 1 (primary protected)",
            "race": {"classes": race_cats, "note": "secondary stress audit only"}},
        "removed_columns": REMOVED, "kept_columns": NUMERIC + CATEGORICAL,
        "input_dim": int(X.shape[1]), "numeric_standardisation_fit_role": "defense_train", "numeric_norm": norm,
        "categories": "fixed loader category sets (one-hot)",
        "shortcut_audit": {
            "relationship_husband_wife_rows_share": float(husband_wife.mean()),
            "sex_given_relationship": sex_given_rel,
            "note_relationship": ("relationship in {Husband, Wife} determines SEX for ~"
                                  f"{100 * husband_wife.mean():.0f}% of rows (Husband ~all male, Wife ~all female). "
                                  "relationship is not an equivalent column of sex (its other four values do not determine it), "
                                  "so under the registered rule it is kept as an ordinary permitted predictor; it is the main "
                                  "direct sex signal the defenses must remove. Disclosed, not repaired."),
            "occupation_group_given_workclass_max_share": occ_by_wc,
            "note_occupation": ("occupation (the target source) is removed; no remaining column or combination is a recoding "
                                "of occupation_group (no workclass value fixes the class); education and workclass are "
                                "ordinary correlated predictors."),
            "education_vs_education_num": "education-num is a numeric code of education (duplicate encodings of one permitted fact; kept)"},
        "roles": {r: {"rows": int(len(ix)), "groups": int(len(np.unique(W['unit'][ix]))),
                      "row_id_sha256": sha_arr(np.sort(W['row_id'][ix]).astype(np.int64))} for r, ix in idx.items()},
        "role_semantics": {
            "defense_train": "encoders, heads, critics, LEACE maps, FARE trees (oar defense_fit minus head holdout)",
            "defense_val": "warm-start/head early stopping and defense-side validation (oar head holdout)",
            "attacker_fit": "inner selection attackers and final audit attackers: fitting",
            "attacker_val": "inner selection (utility gates + inner recovery) and final audit attacker selection",
            "assessment": "single outer scoring after SELECTION_LOCK (already-used outer pool; development evidence)",
            "cert": "FARE certification carve-out (oar-cert-v1)",
            "excluded_exposure": "test records equal to a training record (excluded from every scored role)",
            "excluded_dup": "train rows duplicating a test-role record (never used)"},
        "resampling_unit": "exact-record group (unit = de-duplicated canon_key); Adult has no household identifiers",
        "private_inputs_npz": "~/PCRL_eval_cache_private/jcv_v1/inputs/adult_jcv.npz",
        "private_inputs_sha256": sha_file(INPUTS),
        "X_sha256": sha_arr(X), "fits_performed": 0}
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "DATA_ADMISSION.json").write_text(json.dumps(adm, indent=1) + "\n")
    return adm


def load():
    z = np.load(INPUTS, allow_pickle=False)
    D = {k: z[k] for k in z.files}
    D["idx"] = {r: np.flatnonzero(D["role"] == r) for r in np.unique(D["role"])}
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    return D


if __name__ == "__main__":
    a = build()
    print(json.dumps({k: a[k] for k in ("input_dim", "roles")}, indent=1))
    print(json.dumps(a["shortcut_audit"]["note_relationship"]))
