"""A2: deterministic regeneration of the PCRL train and test splits (Adult, HMDA) with the b96c412 loaders.

No model is fitted. Private outputs (row level, outside git) under ~/PCRL_eval_cache_private/bench_v1/inputs/:
  <ds>_features.npz  row_id, split, features (float32, exactly what the b96c412 dataset object yields)
  <ds>_rows.npz      row_id, split, record_key, canon_key, unit, test_role, <sensitive attrs>, task_<task>
Repo output: notes/admission/rows_provenance.json (counts, hashes, match flags only).

Row ids: test rows keep their test-split position (0..n_test-1, = the pilot's row_id for Adult); train rows get
n_test + train position.

Record key (pilot definition, prepare_pilot_adult_s0.py): sha256('|'.join(map(str,row))).hexdigest()[:20] over the
raw row incl. labels -- Adult: AdultDataset.raw_df (post-dropna, pre-encoding); HMDA: the row of
prepare_hmda.load_filtered(hmda_2023_ca.csv) (the processed record's raw source; 17 kept columns incl.
action_taken and loan_amount).
canon_key: Adult only -- the same hash after removing the trailing '.' that adult.test appends to the income
label ('>50K.' vs '>50K'); without it no train row could ever match a test row. HMDA canon_key = record_key.
Unit = de-duplicated record by canon_key, shared within and across splits (no households invented).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import (INPUTS, NOTES, PCRL_DATA, ROLE_SALT, ensure_export, record_key, role_of,  # noqa
                              sha256_array, sha256_file, write_json)

PREP_INPUT_SHA = {  # from combined_evaluation_preparation_v1/notes/artifacts/row_index_verification.json
    "adult/adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult/adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
    "hmda_raw/hmda_2023_ca.csv": "dcd9c1a2b7e29cbd933debb10c4642be7823d219a728d39396204eb3719088e3",
    "hmda_processed/train.npz": "a68c40dedc2e47adb3f2c553c4da13e7f0dd5ee0fd6e6047cf6e43ea78ac6a35",
    "hmda_processed/val.npz": "07791907e9a8d73a86b62be8075e63cc84db6763cbd4e1984e4faa380c8a31df",
    "hmda_processed/test.npz": "e11679cf92fdf9b3723adcfc3b7032e851125b8468fd5d42fbe7d01dac204cc1",
}
PREP_IDX_SHA = {"adult_train_raw_row_index": "61cabbc61e12524ed0a05729915e805dccde87d2d962064b2ff7caaab0e58a9a",
                "adult_test_raw_row_index": "e603c71c03c71f25c58cfa7bb4ac1468a931f3f996fbd4a0b9d3e7ad93cbc19c",
                "hmda_train_filtered_row_index": "e07ba5cdb29f49c53b8e6e8a25bc3831515dbbcd6da4715de0679cdc156ed419"}


def idx_sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a).astype(np.int64)).tobytes()).hexdigest()


def assemble(ds, n_te, n_tr, rk_te, rk_tr, ck_te, ck_tr, attrs, tasks, X_te, X_tr):
    row_id = np.arange(n_te + n_tr, dtype=np.int64)
    split = np.array(["test"] * n_te + ["train"] * n_tr)
    rk = np.concatenate([rk_te, rk_tr])
    ck = np.concatenate([ck_te, ck_tr])
    _, unit = np.unique(ck, return_inverse=True)
    test_role = np.array([role_of(k, ROLE_SALT[ds]) for k in rk_te] + [""] * n_tr)
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.savez(INPUTS / f"{ds}_features.npz", row_id=row_id, split=split,
             features=np.concatenate([X_te, X_tr]).astype(np.float32))
    np.savez(INPUTS / f"{ds}_rows.npz", row_id=row_id, split=split, record_key=rk, canon_key=ck,
             unit=unit.astype(np.int64), test_role=test_role, **attrs, **{f"task_{k}": v for k, v in tasks.items()})
    te, tr = split == "test", split == "train"
    ck_te_set = set(ck_te.tolist())
    overlap = np.array([k in ck_te_set for k in ck_tr])
    return {
        "n_test": int(n_te), "n_train": int(n_tr),
        "n_units_total": int(unit.max() + 1),
        "n_units_test": int(len(np.unique(unit[te]))), "n_units_train": int(len(np.unique(unit[tr]))),
        "duplicate_rows_within_test": int(n_te - len(np.unique(ck_te))),
        "duplicate_rows_within_train": int(n_tr - len(np.unique(ck_tr))),
        "train_rows_whose_key_occurs_in_test": int(overlap.sum()),
        "train_units_shared_with_test": int(len(set(ck_tr[overlap].tolist()))),
        "test_rows_per_role": {r: int((test_role[te] == r).sum()) for r in ("attacker_fit", "attacker_val",
                                                                            "assessment")},
        "test_units_per_role": {r: int(len(np.unique(unit[te][test_role[te] == r])))
                                for r in ("attacker_fit", "attacker_val", "assessment")},
        "record_key_partition_equals_canon_partition_within_test":
            bool(len(np.unique(rk_te)) == len(np.unique(ck_te)) == len(np.unique(np.char.add(rk_te, ck_te)))),
        "features_sha256": sha256_file(INPUTS / f"{ds}_features.npz"),
        "rows_sha256": sha256_file(INPUTS / f"{ds}_rows.npz"),
        "record_key_array_sha256": sha256_array(rk),
    }


def adult():
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    for f in ("adult.data", "adult.test"):
        if not (PCRL_DATA / "adult" / f).exists():
            raise SystemExit(f"missing {f}: the historical loader would silently synthesise data; refusing")
    P = get_adult_purposes()
    tr = AdultDataset(purposes=P, root=str(PCRL_DATA), split="train", download=False)
    te = AdultDataset(purposes=P, root=str(PCRL_DATA), split="test", download=False, norm_stats=tr.norm_stats)
    # raw-row identity (as regenerate_row_indices.py)
    raw = tr._load_csv(PCRL_DATA / "adult" / "adult.data")
    np.random.seed(42)
    perm = np.random.permutation(len(raw))
    k = int(0.8 * len(raw))
    tr_raw = perm[:k][~raw.iloc[perm[:k]].isna().any(axis=1).to_numpy()]
    te_raw_df = tr._load_csv(PCRL_DATA / "adult" / "adult.test", skip_first=True)
    te_raw = np.flatnonzero(~te_raw_df.isna().any(axis=1).to_numpy())
    # the loader's raw_df rows equal the raw file rows at those indices
    raw_rows_equal = bool(raw.iloc[tr_raw].reset_index(drop=True).equals(tr.raw_df)
                          and te_raw_df.iloc[te_raw].reset_index(drop=True).equals(te.raw_df))

    def keys(df):
        rk = np.array([record_key(r) for r in df.itertuples(index=False)])
        d2 = df.copy()
        d2["income"] = d2["income"].astype(str).str.rstrip(".")
        ck = np.array([record_key(r) for r in d2.itertuples(index=False)])
        return rk, ck

    rk_te, ck_te = keys(te.raw_df)
    rk_tr, ck_tr = keys(tr.raw_df)
    attrs = {a: np.concatenate([te.sensitive_attrs[a].numpy(), tr.sensitive_attrs[a].numpy()]).astype(np.int64)
             for a in te.sensitive_attrs}
    tasks = {t: np.concatenate([te.task_labels[t].numpy(), tr.task_labels[t].numpy()]).astype(np.int64)
             for t in te.task_labels}
    rec = assemble("adult", len(te), len(tr), rk_te, rk_tr, ck_te, ck_tr, attrs, tasks,
                   te.features.numpy(), tr.features.numpy())
    rec.update({
        "loader": "b96c412 pcrl.data.adult.AdultDataset(split=train) / (split=test, norm_stats=train.norm_stats)",
        "income_label_strings_test": sorted(set(te.raw_df["income"].astype(str))),
        "income_label_strings_train": sorted(set(tr.raw_df["income"].astype(str))),
        "train_raw_row_index_sha256": idx_sha(tr_raw),
        "train_raw_row_index_matches_preparation": idx_sha(tr_raw) == PREP_IDX_SHA["adult_train_raw_row_index"],
        "test_raw_row_index_matches_preparation": idx_sha(te_raw) == PREP_IDX_SHA["adult_test_raw_row_index"],
        "loader_raw_df_equals_raw_file_rows": raw_rows_equal,
        "input_dim": int(te.features.shape[1]),
        "sensitive_attrs": list(te.sensitive_attrs), "task_labels": list(te.task_labels),
        "sensitive_income_equals_task_income": bool(np.array_equal(attrs.get("income"), tasks.get("income")))
        if "income" in attrs else None,
        "norm_stats_train": {c: [float(v[0]), float(v[1])] for c, v in tr.norm_stats.items()},
    })
    return rec


def hmda():
    import prepare_hmda as ph
    from pcrl.data.hmda import HMDADataset, get_hmda_purposes
    P = get_hmda_purposes()
    ds = {sp: HMDADataset(purposes=P, root=str(PCRL_DATA), split=sp) for sp in ("train", "val", "test")}
    df = ph.load_filtered(PCRL_DATA / "hmda_raw" / "hmda_2023_ca.csv")
    tri, vai, tei = ph.split_indices(len(df), seed=42)
    mask = np.zeros(len(df), bool)
    mask[tri] = True
    X, _, norm = ph.encode_features(df, mask)
    T, A, lstats = ph.encode_labels(df, mask)
    eq = {}
    for sp, ix in (("train", tri), ("val", vai), ("test", tei)):
        d = ds[sp]
        eq[sp] = {"n": int(len(ix)), "features_equal": bool(np.array_equal(X[ix].astype(np.float32),
                                                                             d.features.numpy())),
                  **{f"task_{t}_equal": bool(np.array_equal(T[t][ix], d.task_labels[t].numpy())) for t in T},
                  **{f"attr_{a}_equal": bool(np.array_equal(A[a][ix], d.sensitive_attrs[a].numpy())) for a in A}}
    import json
    meta = json.loads((PCRL_DATA / "hmda_processed" / "metadata.json").read_text())
    norm_eq = all(abs(meta["norm_stats"][k][q] - norm[k][q]) == 0 for k in norm for q in ("mean", "std"))
    lab_eq = (meta["label_stats"]["loan_amount_band_cutoffs"] == lstats["loan_amount_band_cutoffs"]
              and meta["label_stats"]["tract_denial_median_rate"] == lstats["tract_denial_median_rate"])
    rk_all = np.array([record_key(r) for r in df.itertuples(index=False)])
    rk_te, rk_tr = rk_all[tei], rk_all[tri]
    te, tr = ds["test"], ds["train"]
    attrs = {a: np.concatenate([te.sensitive_attrs[a].numpy(), tr.sensitive_attrs[a].numpy()]).astype(np.int64)
             for a in te.sensitive_attrs}
    tasks = {t: np.concatenate([te.task_labels[t].numpy(), tr.task_labels[t].numpy()]).astype(np.int64)
             for t in te.task_labels}
    rec = assemble("hmda", len(te), len(tr), rk_te, rk_tr, rk_te, rk_tr, attrs, tasks,
                   te.features.numpy(), tr.features.numpy())
    rec.update({
        "loader": "b96c412 pcrl.data.hmda.HMDADataset(split=train|test) over data/hmda_processed; record keys from "
                  "b96c412 experiments/prepare_hmda.py load_filtered(data/hmda_raw/hmda_2023_ca.csv)",
        "n_filtered": int(len(df)), "filtered_columns": list(df.columns),
        "train_filtered_row_index_sha256": idx_sha(tri),
        "train_filtered_row_index_matches_preparation": idx_sha(tri) == PREP_IDX_SHA["hmda_train_filtered_row_index"],
        "regenerated_vs_local_processed": eq,
        "metadata_norm_stats_equal_regenerated": bool(norm_eq),
        "metadata_label_stats_equal_regenerated": bool(lab_eq),
        "input_dim": int(te.features.shape[1]),
        "sensitive_attrs": list(te.sensitive_attrs), "task_labels": list(te.task_labels),
    })
    return rec


def main():
    ensure_export()
    out = {"input_sha256": {p: sha256_file(PCRL_DATA / p) for p in PREP_INPUT_SHA},
           "fits_performed": 0}
    out["input_sha256_matches_preparation"] = {p: out["input_sha256"][p] == h for p, h in PREP_INPUT_SHA.items()}
    out["adult"] = adult()
    out["hmda"] = hmda()
    import numpy, pandas
    out["versions"] = {"numpy": numpy.__version__, "pandas": pandas.__version__}
    write_json(NOTES / "rows_provenance.json", out)
    for d in ("adult", "hmda"):
        r = out[d]
        print(d, {k: r[k] for k in ("n_test", "n_train", "n_units_total", "duplicate_rows_within_test",
                                     "duplicate_rows_within_train", "train_rows_whose_key_occurs_in_test",
                                     "test_rows_per_role")})


if __name__ == "__main__":
    main()
