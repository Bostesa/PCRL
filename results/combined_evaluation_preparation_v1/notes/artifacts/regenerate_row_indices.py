"""Regenerate the row-index arrays of the candidate frozen-artifact interfaces.

Role: artifact/provenance (combined_evaluation_preparation_v1). No model is fit,
no checkpoint is run. This only re-derives which rows each role uses, from the
already-used local development data, and checks the derivation against stored
aligned label vectors where they exist (sequence equality, not length).

Inputs (paths are arguments so nothing private is hard-coded in the repo):
  --pcrl-snapshot  dir containing a `pcrl/` package snapshot (git archive of a
                   PCRL ref whose pcrl/data/{adult,hmda,diabetes,base}.py blobs
                   equal ee8218d/4474317/8cadb2e/74c6ab1 -- identical on every ref
                   since be88a825e) and experiments/{prepare_hmda,preprocess_diabetes}.py
                   (blob d26adaf / e3ac31107).
  --pcrl-data      PCRL data root (adult/, hmda_processed/, hmda_raw/, diabetes/,
                   diabetes_processed/).
  --dg-scores      optional: durable-guarantees analysis/ dir holding
                   tpr59_scores/{adult_noise_s0,hmda_noise_s8}.npz (drive members,
                   sha256-verified against tree-durable-guarantees inventory).
  --out            JSON output (counts, sha256 of index arrays, match flags only;
                   no individual-level values are written).

Run: .venv/bin/python regenerate_row_indices.py --pcrl-snapshot S --pcrl-data D --dg-scores A --out O
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.int64)).tobytes()).hexdigest()


def counts(y: np.ndarray) -> list[int]:
    return [int(c) for c in np.bincount(y)]


def attacker_split(y: np.ndarray, seed: int):
    """durable-guarantees convention: sklearn train_test_split(test_size=0.25,
    random_state=seed, stratify=y) on rows in loader order (shuffle=False).
    Sources: dg@956f5c8 utils/battery.py:50-55, experiments/honest_reaudit.py:79-80,
    experiments/run_tpr_failing59.py:89-90,110-111, run_tpr_extension.py:250-251."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)
    return tr, te


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcrl-snapshot", required=True)
    ap.add_argument("--pcrl-data", required=True)
    ap.add_argument("--dg-scores", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    snap, data = Path(a.pcrl_snapshot), Path(a.pcrl_data)
    sys.path.insert(0, str(snap))
    sys.path.insert(0, str(snap / "experiments"))
    out: dict = {"seeds_attacker_split": [0, 1, 2], "interfaces": {}}
    import sklearn
    out["sklearn_version"] = sklearn.__version__
    out["numpy_version"] = np.__version__
    out["input_sha256"] = {p: file_sha(data / p) for p in [
        "adult/adult.data", "adult/adult.test", "hmda_raw/hmda_2023_ca.csv",
        "hmda_processed/train.npz", "hmda_processed/val.npz", "hmda_processed/test.npz",
        "diabetes/diabetic_data.csv", "diabetes_processed/train.npz",
        "diabetes_processed/val.npz", "diabetes_processed/test.npz"]}

    # ---------------- Adult (PCRL AdultDataset) ----------------------------
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    import pandas as pd
    P = get_adult_purposes()
    tr = AdultDataset(purposes=P, root=str(data), split="train", download=False)
    va = AdultDataset(purposes=P, root=str(data), split="val", download=False,
                      norm_stats=tr.norm_stats)
    te = AdultDataset(purposes=P, root=str(data), split="test", download=False,
                      norm_stats=tr.norm_stats)
    # raw-row identity: permutation of adult.data rows (seed 42), first 80% =
    # train, then dropna AFTER the split (pcrl/data/adult.py:196-204,311).
    raw = tr._load_csv(data / "adult" / "adult.data")
    np.random.seed(42)
    perm = np.random.permutation(len(raw))
    k = int(0.8 * len(raw))
    tr_raw = perm[:k][~raw.iloc[perm[:k]].isna().any(axis=1).to_numpy()]
    va_raw = perm[k:][~raw.iloc[perm[k:]].isna().any(axis=1).to_numpy()]
    te_raw_df = tr._load_csv(data / "adult" / "adult.test", skip_first=True)
    te_raw = np.flatnonzero(~te_raw_df.isna().any(axis=1).to_numpy())
    sex = tr.sensitive_attrs["sex"].numpy()
    inc = tr.task_labels["income"].numpy()
    ad = {"pcrl_split_rule": "adult.data rows permuted np.random.seed(42); first int(0.8*32561)=26048 -> train, rest -> val; dropna after split; adult.test (header skipped, dropna) -> test",
          "n_raw_adult_data": int(len(raw)), "n_train_pre_dropna": k,
          "n_train": len(tr), "n_val": len(va), "n_test": len(te),
          "train_raw_row_index_sha256": sha(tr_raw), "val_raw_row_index_sha256": sha(va_raw),
          "test_raw_row_index_sha256": sha(te_raw),
          "train_raw_rows_consistent": bool(len(tr_raw) == len(tr)),
          "sex_counts_train(F,M)": counts(sex), "income_counts_train": counts(inc),
          "sex_counts_test(F,M)": counts(te.sensitive_attrs["sex"].numpy()),
          "attacker_splits": {}}
    for s in (0, 1, 2):
        a_tr, a_te = attacker_split(sex, s)
        ad["attacker_splits"][s] = {"fit_n": len(a_tr), "assess_n": len(a_te),
                                    "fit_sex_counts": counts(sex[a_tr]),
                                    "assess_sex_counts": counts(sex[a_te]),
                                    "fit_idx_sha256": sha(a_tr), "assess_idx_sha256": sha(a_te)}
    out["interfaces"]["adult_train_dg"] = ad

    # ---------------- HMDA (processed npz = row identity) -------------------
    h = {}
    for sp in ("train", "val", "test"):
        z = np.load(data / "hmda_processed" / f"{sp}.npz")
        h[sp] = {k2: z[k2] for k2 in z.files}
    race = h["train"]["attr_race"].astype(np.int64)
    hm = {"pcrl_split_rule": "prepare_hmda.py: API-filtered CA-2023 CSV -> load_filtered (row filters, reset_index) -> RandomState(42).permutation(n); 70/15/15 (experiments/prepare_hmda.py:431-441,475)",
          "n_train": int(len(race)), "n_val": int(len(h["val"]["attr_race"])),
          "n_test": int(len(h["test"]["attr_race"])),
          "race_counts_train": counts(race),
          "loan_decision_counts_train": counts(h["train"]["task_loan_decision"].astype(np.int64)),
          "loan_amount_band_counts_train": counts(h["train"]["task_loan_amount_band"].astype(np.int64)),
          "race_counts_test": counts(h["test"]["attr_race"].astype(np.int64)),
          "attacker_splits": {}}
    for s in (0, 1, 2):
        a_tr, a_te = attacker_split(race, s)
        hm["attacker_splits"][s] = {"fit_n": len(a_tr), "assess_n": len(a_te),
                                    "fit_race_counts": counts(race[a_tr]),
                                    "assess_race_counts": counts(race[a_te]),
                                    "fit_idx_sha256": sha(a_tr), "assess_idx_sha256": sha(a_te)}
    # regenerate processed arrays from local raw CSV with the committed code
    try:
        import prepare_hmda as ph
        df = ph.load_filtered(data / "hmda_raw" / "hmda_2023_ca.csv")
        tri, vai, tei = ph.split_indices(len(df), seed=42)
        mask = np.zeros(len(df), bool); mask[tri] = True
        X, _, _ = ph.encode_features(df, mask)
        same = {sp: bool(np.array_equal(X[ix].astype(np.float32), h[sp]["features"]))
                for sp, ix in (("train", tri), ("val", vai), ("test", tei))}
        hm["regenerated_from_raw"] = {"n_filtered": int(len(df)), "features_equal": same,
                                      "train_filtered_row_index_sha256": sha(tri)}
    except Exception as e:  # report, do not hide
        hm["regenerated_from_raw"] = {"error": f"{type(e).__name__}: {e}"}
    out["interfaces"]["hmda_train_dg"] = hm

    # ---------------- Diabetes (PCRL NeurIPS R4-R7) ------------------------
    dz = {sp: np.load(data / "diabetes_processed" / f"{sp}.npz") for sp in ("train", "val", "test")}
    db = {"pcrl_split_rule": "preprocess_diabetes.py: drop missing age/diag_1; drop_duplicates(patient_nbr, keep=first) -> one encounter per patient; drop unknown gender; per-class (readmission) RandomState(42) shuffles 70/15/15 (experiments/preprocess_diabetes.py:220-275)",
          "n": {sp: int(len(dz[sp]["race"])) for sp in dz},
          "grouping_unit": "patient_nbr (deduplicated before split: each patient in exactly one row/split)"}
    try:
        import pandas as pd
        raw = pd.read_csv(data / "diabetes" / "diabetic_data.csv", na_values=["?"], low_memory=False)
        r1 = raw[raw["age"].notna() & raw["diag_1"].notna()]
        r2 = r1.drop_duplicates(subset=["patient_nbr"], keep="first")
        db["raw_rows"] = int(len(raw)); db["after_dedup_patient"] = int(len(r2))
        db["sum_processed"] = int(sum(db["n"].values()))
    except Exception as e:
        db["raw_check_error"] = f"{type(e).__name__}: {e}"
    out["interfaces"]["diabetes_pcrl"] = db

    # ---------------- check against stored dg aligned label vectors --------
    if a.dg_scores:
        A = Path(a.dg_scores)
        chk = {}
        for name, y, f in (("adult_sex", sex, "tpr59_scores/adult_noise_s0.npz"),
                           ("hmda_race", race, "tpr59_scores/hmda_noise_s8.npz")):
            p = A / f
            if not p.exists():
                chk[name] = "score file absent"; continue
            z = np.load(p)
            r = {"file": f, "file_sha256": file_sha(p), "keys": sorted(z.files),
                 "row_index_key_present": any("idx" in k or "index" in k or "row" in k for k in z.files)}
            for s in (0, 1, 2):
                _, a_te = attacker_split(y, s)
                stored = z[f"y_ps{s}"].astype(np.int64)
                r[f"seed{s}_len_equal"] = bool(len(stored) == len(a_te))
                r[f"seed{s}_sequence_equal"] = bool(len(stored) == len(a_te) and np.array_equal(stored, y[a_te]))
            chk[name] = r
        out["stored_vector_alignment_check"] = chk

    Path(a.out).write_text(json.dumps(out, indent=1, default=int))
    print(json.dumps(out, indent=1, default=int)[:6000])


if __name__ == "__main__":
    main()
