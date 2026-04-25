#!/usr/bin/env python3
"""Preprocess UCI Diabetes 130-US dataset for PCRL.

Downloads archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip,
reads diabetic_data.csv, and emits preprocessed .npz per split to
data/diabetes_processed/.

Preprocessing:
- Drop high-missing cols: weight, payer_code, medical_specialty.
- Encode race (5 cats, "?" -> Other), gender (binary), age (10 decade buckets).
- Group diag_1 into 9 ICD-9 categories (Strack et al. 2014):
  circulatory, respiratory, digestive, diabetes, injury, musculoskeletal,
  genitourinary, neoplasms, other.
- Features: z-normalized numericals + one-hot categoricals.
- Targets: readmission_outcome (<30d binary), primary_diagnosis_category (9-class),
  medication_change_outcome (binary from 'change' column).
- 70/15/15 stratified by readmission_outcome with seed=42.

Outputs per split:
  data/diabetes_processed/{train,val,test}.npz with:
    features, readmission_outcome, primary_diagnosis_category,
    medication_change_outcome, race, gender, age_bucket.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

RAW_DIR = Path("data/diabetes")
OUT_DIR = Path("data/diabetes_processed")

HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}

RACE_ORDER = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
RACE_MAP = {r: i for i, r in enumerate(RACE_ORDER)}

GENDER_ORDER = ["Female", "Male"]
GENDER_MAP = {g: i for i, g in enumerate(GENDER_ORDER)}

DIAG_ORDER = [
    "circulatory", "diabetes", "digestive", "injury", "musculoskeletal",
    "respiratory", "genitourinary", "neoplasms", "other",
]
DIAG_MAP = {d: i for i, d in enumerate(DIAG_ORDER)}

NUMERICAL = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

# Categorical integer-coded IDs (treat as categorical, one-hot)
CAT_INT = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# 23 medication columns (UCI data dict)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

MED_LEVELS = ["No", "Steady", "Up", "Down"]
MED_MAP = {v: i for i, v in enumerate(MED_LEVELS)}

# Additional categoricals
A1C_LEVELS = ["None", "Norm", ">7", ">8"]
A1C_MAP = {v: i for i, v in enumerate(A1C_LEVELS)}

GLU_LEVELS = ["None", "Norm", ">200", ">300"]
GLU_MAP = {v: i for i, v in enumerate(GLU_LEVELS)}


def icd9_to_category(code) -> str:
    if code is None or pd.isna(code):
        return "other"
    s = str(code).strip()
    if not s or s == "?":
        return "other"
    # V/E codes -> other
    if s[0] in ("V", "E"):
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"
    # Diabetes: 250.xx (250 <= v < 251)
    if 250.0 <= v < 251.0:
        return "diabetes"
    iv = int(v)
    if (390 <= iv <= 459) or iv == 785:
        return "circulatory"
    if (460 <= iv <= 519) or iv == 786:
        return "respiratory"
    if (520 <= iv <= 579) or iv == 787:
        return "digestive"
    if 800 <= iv <= 999:
        return "injury"
    if 710 <= iv <= 739:
        return "musculoskeletal"
    if (580 <= iv <= 629) or iv == 788:
        return "genitourinary"
    if 140 <= iv <= 239:
        return "neoplasms"
    return "other"


def download_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "diabetic_data.csv"
    if csv_path.exists():
        print(f"  Raw CSV already present at {csv_path}")
        return csv_path
    zip_path = RAW_DIR / "dataset_diabetes.zip"
    print(f"  Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    print(f"  Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    inner = RAW_DIR / "dataset_diabetes" / "diabetic_data.csv"
    if inner.exists():
        inner.rename(csv_path)
    else:
        raise RuntimeError(f"diabetic_data.csv not found after extract; {list(RAW_DIR.rglob('*.csv'))}")
    return csv_path


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (features_matrix, column_names). Uses FIXED category sets so
    splits share identical feature column ordering."""
    blocks: list[np.ndarray] = []
    names: list[str] = []

    # Numericals -> z-normalized using this split's statistics.
    # We will z-normalize w.r.t. TRAIN stats after the split.  Here we just
    # return raw numericals; caller handles normalization.
    for col in NUMERICAL:
        blocks.append(df[col].values.astype(np.float32).reshape(-1, 1))
        names.append(f"num::{col}")

    # age_bucket one-hot (10 cats)
    age_vals = df["age_bucket"].values.astype(int)
    oh = np.zeros((len(age_vals), 10), dtype=np.float32)
    oh[np.arange(len(age_vals)), age_vals] = 1.0
    blocks.append(oh)
    names += [f"age_bucket::{i}" for i in range(10)]

    # CAT_INT one-hot with fixed domain taken from train later; just return
    # integer codes here and let caller align.  For now, one-hot with the
    # fixed set observed in the full dataframe (passed in via df which is
    # the FULL df before splitting — see main()).
    for col in CAT_INT:
        uniq = sorted(int(v) for v in df[col].dropna().unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        codes = df[col].map(mapping).fillna(0).astype(int).values
        oh = np.zeros((len(codes), len(uniq)), dtype=np.float32)
        oh[np.arange(len(codes)), codes] = 1.0
        blocks.append(oh)
        names += [f"{col}::{v}" for v in uniq]

    # Medications: one-hot 4 levels each
    for col in MED_COLS:
        codes = df[col].map(MED_MAP).fillna(0).astype(int).values
        oh = np.zeros((len(codes), 4), dtype=np.float32)
        oh[np.arange(len(codes)), codes] = 1.0
        blocks.append(oh)
        names += [f"{col}::{lvl}" for lvl in MED_LEVELS]

    # A1Cresult one-hot (4)
    codes = df["A1Cresult"].fillna("None").map(A1C_MAP).fillna(0).astype(int).values
    oh = np.zeros((len(codes), 4), dtype=np.float32)
    oh[np.arange(len(codes)), codes] = 1.0
    blocks.append(oh); names += [f"A1Cresult::{lvl}" for lvl in A1C_LEVELS]

    # max_glu_serum one-hot (4)
    codes = df["max_glu_serum"].fillna("None").map(GLU_MAP).fillna(0).astype(int).values
    oh = np.zeros((len(codes), 4), dtype=np.float32)
    oh[np.arange(len(codes)), codes] = 1.0
    blocks.append(oh); names += [f"max_glu_serum::{lvl}" for lvl in GLU_LEVELS]

    # diabetesMed binary
    dm = (df["diabetesMed"] == "Yes").astype(np.float32).values.reshape(-1, 1)
    blocks.append(dm); names.append("diabetesMed::Yes")

    X = np.concatenate(blocks, axis=1)
    return X, names


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = download_raw()
    print(f"  Reading {csv_path}")
    df = pd.read_csv(csv_path, na_values=["?"], low_memory=False)
    print(f"  Raw: {len(df)} rows, {df.shape[1]} cols")

    # ── Drop high-missing ───────────────────────────────────────────────
    df = df.drop(columns=[c for c in HIGH_MISSING_COLS if c in df.columns])
    print(f"  After drop high-missing: {df.shape[1]} cols")

    # ── Required non-null: age, diag_1, race can be mapped ─────────────
    df = df[df["age"].notna() & df["diag_1"].notna()].reset_index(drop=True)
    print(f"  After drop missing age/diag_1: {len(df)} rows")

    # ── Keep only 1 encounter per patient (first) to avoid leakage ─────
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first").reset_index(drop=True)
    print(f"  After dedup by patient_nbr: {len(df)} rows")

    # ── Encode sensitive attributes ────────────────────────────────────
    df["race"] = df["race"].fillna("Other")
    df.loc[~df["race"].isin(RACE_ORDER), "race"] = "Other"
    df["race_code"] = df["race"].map(RACE_MAP).astype(int)

    df["gender_code"] = df["gender"].map(GENDER_MAP)
    df = df[df["gender_code"].notna()].reset_index(drop=True)
    df["gender_code"] = df["gender_code"].astype(int)

    df["age_bucket"] = df["age"].map(AGE_MAP).astype(int)

    # ── Targets ─────────────────────────────────────────────────────────
    df["readmission_outcome"] = (df["readmitted"] == "<30").astype(int)
    df["medication_change_outcome"] = (df["change"] == "Ch").astype(int)
    df["primary_diagnosis_category_str"] = df["diag_1"].apply(icd9_to_category)
    df["primary_diagnosis_category"] = df["primary_diagnosis_category_str"].map(DIAG_MAP).astype(int)

    # ── Fill remaining NaNs in numerical/categorical used for features ─
    for col in NUMERICAL:
        df[col] = df[col].fillna(df[col].median())
    for col in CAT_INT:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    for col in MED_COLS + ["A1Cresult", "max_glu_serum", "diabetesMed"]:
        df[col] = df[col].fillna("No" if col in MED_COLS else ("None" if col in ("A1Cresult", "max_glu_serum") else "No"))

    # ── Build features on FULL df (for consistent one-hot categories) ──
    print("  Building features...")
    X_full, feat_names = build_features(df)
    print(f"  Features: {X_full.shape[1]} dims")

    # ── 70/15/15 split stratified by readmission_outcome, seed=42 ──────
    rng = np.random.RandomState(42)
    y = df["readmission_outcome"].values
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def split_arr(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    p_tr, p_va, p_te = split_arr(idx_pos)
    n_tr, n_va, n_te = split_arr(idx_neg)
    tr_idx = np.concatenate([p_tr, n_tr]); rng.shuffle(tr_idx)
    va_idx = np.concatenate([p_va, n_va]); rng.shuffle(va_idx)
    te_idx = np.concatenate([p_te, n_te]); rng.shuffle(te_idx)
    print(f"  Split: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

    # ── Z-normalize the numerical block using train stats ─────────────
    n_num = len(NUMERICAL)
    num_train = X_full[tr_idx, :n_num]
    mu = num_train.mean(axis=0)
    sigma = num_train.std(axis=0) + 1e-8
    X_full[:, :n_num] = (X_full[:, :n_num] - mu) / sigma
    print(f"  Numerical z-normalized (mu shape {mu.shape})")

    # ── Save ──────────────────────────────────────────────────────────
    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        path = OUT_DIR / f"{name}.npz"
        np.savez_compressed(
            path,
            features=X_full[idx].astype(np.float32),
            readmission_outcome=y[idx].astype(np.int64),
            primary_diagnosis_category=df["primary_diagnosis_category"].values[idx].astype(np.int64),
            medication_change_outcome=df["medication_change_outcome"].values[idx].astype(np.int64),
            race=df["race_code"].values[idx].astype(np.int64),
            gender=df["gender_code"].values[idx].astype(np.int64),
            age_bucket=df["age_bucket"].values[idx].astype(np.int64),
        )
        print(f"  Wrote {path} ({X_full[idx].shape[0]} rows, {X_full[idx].shape[1]} feats)")

    # Feature-name manifest for debugging
    (OUT_DIR / "feature_names.txt").write_text("\n".join(feat_names))
    (OUT_DIR / "meta.txt").write_text(
        f"n_train={len(tr_idx)}\nn_val={len(va_idx)}\nn_test={len(te_idx)}\n"
        f"n_features={X_full.shape[1]}\n"
        f"race_order={RACE_ORDER}\ngender_order={GENDER_ORDER}\n"
        f"diag_order={DIAG_ORDER}\n"
    )
    print(f"\n  Done. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
