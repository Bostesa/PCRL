#!/usr/bin/env python3
"""Download, filter, and preprocess HMDA 2023 California LAR for PCRL.

Source
------
FFIEC CFPB Data Browser dynamic CSV endpoint:
    https://ffiec.cfpb.gov/v2/data-browser-api/view/csv

Pre-filter applied via API:
    states=CA, years=2023, loan_purposes=1 (home purchase),
    lien_statuses=1 (first lien), actions_taken=1,3 (originated or denied),
    construction_methods=1 (site-built only).

Additional row filters applied locally:
    Drop rows with unknown sensitive attributes (race / ethnicity / sex
    not in canonical buckets, age in {"8888","9999"}, debt-to-income
    "NA"/"Exempt"). This guarantees clean integer codes for all sensitive
    attributes used in the empirical compliance audit.

Output (under data/hmda_processed/):
    metadata.json     schema, dims, feature names, normalisation stats
    train.npz         features, task_<name>, attr_<name>
    val.npz           same shape (15% of records)
    test.npz          same shape (15% of records)

Design notes
------------
- ``loan_amount`` is intentionally excluded from features because it would
  trivialise the loan_amount_band classification head used by the
  pricing_analysis purpose. The model must learn to anticipate the band
  from income, DTI, dwelling type, and tract context — exactly the
  setting a regulator cares about.
- ``tract_denial_high`` is computed by aggregating denial counts per
  census tract on the full filtered dataset, then thresholding at the
  median per-tract denial rate. The threshold is fitted before the
  train/val/test split, which is fine because it does not depend on row
  identity within a split — it depends only on the tract a row belongs to.
- ``county_code`` is one-hot encoded over counties observed in training.
  Counties absent from training are encoded as zero across all county
  dummies (acts as an "other" bucket).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.hmda import HMDA_EXPECTED_SCHEMA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("prepare_hmda")

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

FFIEC_CSV_URL = (
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    "?states=CA"
    "&years=2023"
    "&loan_purposes=1"
    "&actions_taken=1,3"
)
# Note: the CFPB API caps "filter criteria" to 2 (states+years are
# always-on and don't count). lien_status=1 and construction_method=1
# are applied locally instead. The endpoint redirects to a pre-built
# CSV at files.ffiec.cfpb.gov; we use curl -L to follow the redirect
# because Python urllib is intermittently blocked at the CDN.

# Columns kept from the raw CSV (everything else is dropped on read).
NEEDED_COLUMNS = [
    "action_taken",
    "loan_amount",
    "loan_type",
    "loan_purpose",
    "lien_status",
    "construction_method",
    "derived_dwelling_category",
    "income",
    "debt_to_income_ratio",
    "applicant_age",
    "applicant_age_above_62",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "county_code",
    "census_tract",
    "tract_minority_population_percent",
]

# Race aggregation: 5 buckets per spec. Buckets aggregate the HMDA
# derived_race string field (5 known categories + "2 or more minority
# races" / "Joint" / "Race Not Available" / "Free Form Text Only").
RACE_TO_INT: dict[str, int] = {
    "White": 0,
    "Black or African American": 1,
    "Asian": 2,
    "American Indian or Alaska Native": 3,
    "Native Hawaiian or Other Pacific Islander": 3,
    "2 or more minority races": 3,
    # 4 = "Other / unknown" — Joint applications and missing-race rows.
    "Joint": 4,
}
# Anything outside the keys above (or NaN) is dropped — see filter loop.
RACE_NUM_CLASSES = 5

# Ethnicity: 2 buckets (Hispanic/Latino vs not).
ETHNICITY_HISPANIC = "Hispanic or Latino"
ETHNICITY_NOT_HISPANIC = "Not Hispanic or Latino"

# Sex: 2 buckets. Keep only Male/Female (drop Joint and Sex Not Available
# so the integer code is unambiguous).
SEX_TO_INT = {"Female": 0, "Male": 1}

# Age above 62 = bucket "65-74" or ">74". (HMDA 2023 reports applicant_age
# as buckets; "55-64" straddles 62 so is excluded from the "above 62" flag.)
AGE_ABOVE_62 = {"65-74", ">74"}
AGE_KNOWN = {"<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"}

# Debt-to-income mapping → numeric midpoint (percent).
DTI_TO_NUMERIC: dict[str, float] = {
    "<20%": 15.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "50%-60%": 55.0,
    ">60%": 65.0,
}
for v in [
    "36", "37", "38", "39", "40", "41", "42", "43",
    "44", "45", "46", "47", "48", "49",
]:
    DTI_TO_NUMERIC[v] = float(v)
# "Exempt" and "NA" are NOT mapped — those rows are dropped.

# Dwelling category → one-hot index.
DWELLING_TO_INT = {
    "Single Family (1-4 Units):Site-Built": 0,
    "Multifamily:Site-Built": 1,
    # Other / unknown / "Free Form" → bucket 2
}
DWELLING_NUM_CLASSES = 3

LOAN_TYPE_NUM_CLASSES = 4  # 1=Conventional, 2=FHA, 3=VA, 4=USDA/RHS


# ─────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────

def download_csv(out_path: Path, retries: int = 3, timeout: int = 1800) -> None:
    """Download the FFIEC filtered CSV via curl (follows CDN redirect).

    The data-browser API redirects to a pre-built CSV at
    ``files.ffiec.cfpb.gov``; that CDN intermittently 403s on Python's
    default urllib User-Agent, so we shell out to curl which follows the
    redirect cleanly with --location and --compressed.
    """
    import subprocess

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        if size_mb > 5:
            log.info(f"raw CSV already at {out_path} ({size_mb:.1f} MB), skipping download")
            return
        log.info(f"existing {out_path} too small ({size_mb:.1f} MB) — re-downloading")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            log.info(f"downloading HMDA 2023 CA LAR  (attempt {attempt + 1}/{retries})")
            log.info(f"  {FFIEC_CSV_URL}")
            t0 = time.time()
            tmp_path = out_path.with_suffix(".csv.tmp")
            result = subprocess.run(
                [
                    "curl",
                    "--silent", "--show-error",
                    "--location",
                    "--compressed",
                    "--max-time", str(timeout),
                    "-A", "PCRL-research/1.0",
                    "-o", str(tmp_path),
                    "-w", "%{http_code}",
                    FFIEC_CSV_URL,
                ],
                check=False, capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr.strip()}")
            http_code = result.stdout.strip()
            if http_code != "200":
                raise RuntimeError(f"HTTP {http_code} (curl rc=0)")
            size_mb = tmp_path.stat().st_size / 1e6
            if size_mb < 5:
                raise RuntimeError(f"downloaded file too small: {size_mb:.1f} MB")
            tmp_path.rename(out_path)
            log.info(f"  downloaded {size_mb:.1f} MB in {time.time() - t0:.0f}s")
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            log.warning(f"  download attempt {attempt + 1} failed: {e}")
            time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError(f"failed to download HMDA CSV after {retries} attempts") from last_err


# ─────────────────────────────────────────────────────────────────────────
# Filter + encode
# ─────────────────────────────────────────────────────────────────────────

def load_filtered(csv_path: Path) -> pd.DataFrame:
    """Load CSV and apply local filters (sensitive attribute completeness)."""
    log.info(f"reading {csv_path}")
    t0 = time.time()
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in NEEDED_COLUMNS,
        dtype={"county_code": "string", "census_tract": "string"},
        low_memory=False,
    )
    log.info(f"  read {len(df):,} rows in {time.time() - t0:.0f}s")
    log.info(f"  columns: {list(df.columns)}")

    # Confirm pre-filters from API actually held.
    df = df[df["action_taken"].isin([1, 3])]
    df = df[df["loan_purpose"] == 1]
    df = df[df["lien_status"] == 1]
    df = df[df["construction_method"] == 1]
    log.info(f"  after API-filter sanity check: {len(df):,} rows")

    # Loan type 1-4 only (the canonical four buckets).
    df = df[df["loan_type"].isin([1, 2, 3, 4])]

    # Race / ethnicity / sex must be in canonical buckets.
    df = df[df["derived_race"].isin(RACE_TO_INT.keys())]
    df = df[df["derived_ethnicity"].isin([ETHNICITY_HISPANIC, ETHNICITY_NOT_HISPANIC])]
    df = df[df["derived_sex"].isin(SEX_TO_INT.keys())]

    # Age must be a known bucket (drop "8888"/"9999"); also require the
    # precomputed above-62 field to be unambiguous.
    df = df[df["applicant_age"].isin(AGE_KNOWN)]
    df = df[df["applicant_age_above_62"].isin({"Yes", "No"})]

    # DTI must map to a numeric midpoint.
    df = df[df["debt_to_income_ratio"].isin(DTI_TO_NUMERIC.keys())]

    # Income must be present and positive (filed in $thousands, integer).
    df = df.dropna(subset=["income", "loan_amount", "tract_minority_population_percent"])
    df = df[df["income"] > 0]
    df = df[df["loan_amount"] > 0]

    # County / census tract must be present.
    df = df.dropna(subset=["county_code", "census_tract"])

    df = df.reset_index(drop=True)
    log.info(f"  after row filters: {len(df):,} rows")
    return df


def encode_features(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    """Encode features. Statistics fit on rows where ``train_mask`` is True.

    Returns
    -------
    X : (N, D) float32 array
    feature_names : list[str]
    norm_stats : dict for metadata
    """
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    norm_stats: dict[str, dict] = {}

    # ── Numeric features (standardised on train) ─────────────────────────
    # income is highly skewed; use log1p before standardising.
    income_log = np.log1p(df["income"].astype(np.float32).values)
    inc_mean = float(income_log[train_mask].mean())
    inc_std = float(income_log[train_mask].std() + 1e-8)
    feature_blocks.append(((income_log - inc_mean) / inc_std).reshape(-1, 1).astype(np.float32))
    feature_names.append("income_log_std")
    norm_stats["income_log_std"] = {"mean": inc_mean, "std": inc_std}

    dti = df["debt_to_income_ratio"].map(DTI_TO_NUMERIC).astype(np.float32).values
    dti_mean = float(dti[train_mask].mean())
    dti_std = float(dti[train_mask].std() + 1e-8)
    feature_blocks.append(((dti - dti_mean) / dti_std).reshape(-1, 1).astype(np.float32))
    feature_names.append("dti_std")
    norm_stats["dti_std"] = {"mean": dti_mean, "std": dti_std}

    tmp = df["tract_minority_population_percent"].astype(np.float32).values
    tmp_mean = float(tmp[train_mask].mean())
    tmp_std = float(tmp[train_mask].std() + 1e-8)
    feature_blocks.append(((tmp - tmp_mean) / tmp_std).reshape(-1, 1).astype(np.float32))
    feature_names.append("tract_minority_pct_std")
    norm_stats["tract_minority_pct_std"] = {"mean": tmp_mean, "std": tmp_std}

    # ── Binary: applicant_age_above_62 ───────────────────────────────────
    # 2023 schema reports this directly as "Yes"/"No"/"NA". Use it.
    age_above = (df["applicant_age_above_62"] == "Yes").astype(np.float32).values
    feature_blocks.append(age_above.reshape(-1, 1))
    feature_names.append("applicant_age_above_62")

    # ── Categorical one-hots ─────────────────────────────────────────────
    # loan_type (4)
    lt = df["loan_type"].astype(int).values
    lt_onehot = np.zeros((len(df), LOAN_TYPE_NUM_CLASSES), dtype=np.float32)
    lt_onehot[np.arange(len(df)), lt - 1] = 1.0
    feature_blocks.append(lt_onehot)
    for i in range(LOAN_TYPE_NUM_CLASSES):
        feature_names.append(f"loan_type_{i + 1}")

    # dwelling category (3)
    dw_codes = df["derived_dwelling_category"].map(
        lambda s: DWELLING_TO_INT.get(s, 2)
    ).astype(int).values
    dw_onehot = np.zeros((len(df), DWELLING_NUM_CLASSES), dtype=np.float32)
    dw_onehot[np.arange(len(df)), dw_codes] = 1.0
    feature_blocks.append(dw_onehot)
    feature_names.extend(["dwelling_single_family", "dwelling_multi", "dwelling_other"])

    # county_code one-hot — counties observed in training only.
    train_counties = sorted(df.loc[train_mask, "county_code"].dropna().unique().tolist())
    county_to_idx = {c: i for i, c in enumerate(train_counties)}
    county_onehot = np.zeros((len(df), len(train_counties)), dtype=np.float32)
    cc = df["county_code"].values
    for row_i, code in enumerate(cc):
        idx = county_to_idx.get(code)
        if idx is not None:
            county_onehot[row_i, idx] = 1.0
    feature_blocks.append(county_onehot)
    for c in train_counties:
        feature_names.append(f"county_{c}")
    log.info(f"  encoded {len(train_counties)} county dummies")

    # race / ethnicity / sex are sensitive AND features (one-hot input).
    race_codes = df["derived_race"].map(RACE_TO_INT).astype(int).values
    race_onehot = np.zeros((len(df), RACE_NUM_CLASSES), dtype=np.float32)
    race_onehot[np.arange(len(df)), race_codes] = 1.0
    feature_blocks.append(race_onehot)
    feature_names.extend([f"race_{i}" for i in range(RACE_NUM_CLASSES)])

    eth_codes = (df["derived_ethnicity"] == ETHNICITY_HISPANIC).astype(int).values
    eth_onehot = np.eye(2, dtype=np.float32)[eth_codes]
    feature_blocks.append(eth_onehot)
    feature_names.extend(["ethnicity_nonhispanic", "ethnicity_hispanic"])

    sex_codes = df["derived_sex"].map(SEX_TO_INT).astype(int).values
    sex_onehot = np.eye(2, dtype=np.float32)[sex_codes]
    feature_blocks.append(sex_onehot)
    feature_names.extend(["sex_female", "sex_male"])

    X = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    log.info(f"  feature matrix: {X.shape}")
    return X, feature_names, norm_stats


def encode_labels(
    df: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Encode task labels and sensitive attributes.

    Returns
    -------
    task_arrays : {name: int64 array}
    attr_arrays : {name: int64 array}
    label_stats : metadata for quintile cutoffs / tract median
    """
    label_stats: dict = {}

    # ── Tasks ────────────────────────────────────────────────────────────
    # loan_decision: 1 = originated, 0 = denied.
    loan_decision = (df["action_taken"] == 1).astype(np.int64).values

    # loan_amount_band: 5-class quintile of loan_amount fitted on training.
    loan_amount = df["loan_amount"].astype(np.float64).values
    cutoffs = np.quantile(loan_amount[train_mask], [0.2, 0.4, 0.6, 0.8])
    bands = np.digitize(loan_amount, cutoffs)  # 0..4
    label_stats["loan_amount_band_cutoffs"] = cutoffs.tolist()

    # tract_denial_high: per-tract denial rate above median (computed over
    # all rows, threshold = median of per-tract denial rate).
    denied = (df["action_taken"] == 3).astype(int).values
    tract = df["census_tract"].values
    tract_df = pd.DataFrame({"tract": tract, "denied": denied})
    rate_by_tract = tract_df.groupby("tract")["denied"].mean()
    median_rate = float(rate_by_tract.median())
    high_tracts = set(rate_by_tract[rate_by_tract > median_rate].index)
    tract_high = np.fromiter(
        (1 if t in high_tracts else 0 for t in tract),
        dtype=np.int64,
        count=len(tract),
    )
    label_stats["tract_denial_median_rate"] = median_rate
    label_stats["num_tracts"] = int(rate_by_tract.shape[0])

    task_arrays = {
        "loan_decision": loan_decision,
        "loan_amount_band": bands.astype(np.int64),
        "tract_denial_high": tract_high,
    }

    # ── Sensitive attributes ─────────────────────────────────────────────
    race = df["derived_race"].map(RACE_TO_INT).astype(np.int64).values
    eth = (df["derived_ethnicity"] == ETHNICITY_HISPANIC).astype(np.int64).values
    sex = df["derived_sex"].map(SEX_TO_INT).astype(np.int64).values

    attr_arrays = {
        "race": race,
        "ethnicity": eth,
        "sex": sex,
    }

    return task_arrays, attr_arrays, label_stats


def split_indices(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 70/15/15 split."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--max-train", type=int, default=300000,
                        help="If filtered set > 500K, randomly sub-sample to this many BEFORE splitting.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "hmda_raw"
    out_dir = data_root / "hmda_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "hmda_2023_ca.csv"
    download_csv(csv_path)

    df = load_filtered(csv_path)
    if len(df) > 500_000:
        log.info(f"  filtered set has {len(df):,} > 500K rows; sub-sampling to {args.max_train:,}")
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(df), size=args.max_train, replace=False)
        df = df.iloc[keep].reset_index(drop=True)
        log.info(f"  sub-sampled to {len(df):,} rows")
    elif len(df) < 50_000:
        log.warning(f"  filtered set has only {len(df):,} rows — proceeding but compliance audit will be noisy")

    # Build the train mask BEFORE encoding so feature stats are computed
    # only on training rows (no leakage of val/test moments into features).
    train_idx, val_idx, test_idx = split_indices(len(df), seed=args.seed)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    log.info(f"  split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    X, feature_names, norm_stats = encode_features(df, train_mask)
    task_arrays, attr_arrays, label_stats = encode_labels(df, train_mask)

    # Save splits.
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, idx in splits.items():
        payload: dict[str, np.ndarray] = {"features": X[idx]}
        for name, arr in task_arrays.items():
            payload[f"task_{name}"] = arr[idx]
        for name, arr in attr_arrays.items():
            payload[f"attr_{name}"] = arr[idx]
        np.savez_compressed(out_dir / f"{split_name}.npz", **payload)
        log.info(f"  wrote {split_name}.npz  N={len(idx)}")

    # Sanity stats — log majority baselines so the compliance audit makes sense.
    for attr_name, arr in attr_arrays.items():
        train_arr = arr[train_idx]
        _, counts = np.unique(train_arr, return_counts=True)
        majority = float(counts.max() / len(train_arr))
        log.info(f"  attr {attr_name}: majority baseline (train) = {majority:.1%}")
    for task_name, arr in task_arrays.items():
        train_arr = arr[train_idx]
        _, counts = np.unique(train_arr, return_counts=True)
        log.info(f"  task {task_name}: train class distribution = {(counts / len(train_arr)).round(3).tolist()}")

    metadata = {
        "schema_version": HMDA_EXPECTED_SCHEMA,
        "source_url": FFIEC_CSV_URL,
        "num_features": int(X.shape[1]),
        "feature_names": feature_names,
        "task_labels": {
            "loan_decision": 2,
            "loan_amount_band": 5,
            "tract_denial_high": 2,
        },
        "sensitive_attrs": {
            "race": RACE_NUM_CLASSES,
            "ethnicity": 2,
            "sex": 2,
        },
        "split_sizes": {"train": int(len(train_idx)),
                        "val": int(len(val_idx)),
                        "test": int(len(test_idx))},
        "norm_stats": norm_stats,
        "label_stats": label_stats,
        "seed": args.seed,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"wrote metadata.json")
    log.info("HMDA preprocessing complete.")


if __name__ == "__main__":
    main()
