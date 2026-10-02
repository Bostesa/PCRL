"""BENCH EFFECTIVE_PROTOCOL: the frozen settings the matched removal benchmark executes.

Authority: results/combined_matched_removal_benchmark_v1/notes/BENCH_DESIGN.md (coordinator design, 2026-10-02).
Every leaf below is CONSUMED by bench.py / bench_infer.py (read through a `Tracked` view; `consumption_report`
lists any leaf that was never read), DESCRIPTIVE (leaf name in DESCRIPTIVE_LEAVES), or listed under "unsupported"
with a reason. `validate_bench_effective` refuses a protocol whose frozen invariants changed. The canonical-JSON
sha256 is pinned in LOCK.json (bench_lock.py).

Recipes are the pilot's verified effective recipes (effective.py: L C-grid, 20 frozen GBT configs, 18 MLP configs,
LRT-A2/A4), extended by: attacker seeds {0,1,2} for the selected recipe, the ignore-rep / ignore-outputs candidates
in the C_rep_plus_clean_out slate, official LEACE (B, C) and the defense_fit role.

Unit ID: <dataset>__s<encoder seed>__<purpose>__<attribute>__<arm>, arm in {A, B, C, D_sigma<sigma:g>_rs<k>}.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .effective import GBT_CONFIGS, LR_C, Tracked, canonical_json, leaf_paths  # noqa: F401  (Tracked re-exported)

BENCH_BRANCH = "research/combined-matched-removal-benchmark-v1"
PILOT_TIP = "661db8dcbba3abd29bd42d6674c2493c77eb47d8"

# Frozen from PCRL b96c412 pcrl/data/{adult,hmda}.py (get_*_purposes list order = purpose index).
DATASETS = {
    "adult": {"purposes": {
        "income_prediction": {"index": 0, "task": "income", "n_task_classes": 2, "disallowed_attrs": ["race", "sex"]},
        "employment_analysis": {"index": 1, "task": "occupation_group", "n_task_classes": 6,
                                "disallowed_attrs": ["race", "age_group", "marital_status"]},
        "education_assessment": {"index": 2, "task": "education_level", "n_task_classes": 4,
                                 "disallowed_attrs": ["sex", "race", "income"]}},
        "attr_dims": {"race": 5, "sex": 2, "age_group": 4, "marital_status": 2, "income": 2}},
    "hmda": {"purposes": {
        "underwriting": {"index": 0, "task": "loan_decision", "n_task_classes": 2,
                         "disallowed_attrs": ["race", "ethnicity"]},
        "pricing_analysis": {"index": 1, "task": "loan_amount_band", "n_task_classes": 5,
                             "disallowed_attrs": ["race", "sex"]},
        "fair_lending_audit": {"index": 2, "task": "tract_denial_high", "n_task_classes": 2,
                               "disallowed_attrs": ["race", "sex"]}},
        "attr_dims": {"race": 5, "ethnicity": 2, "sex": 2}},
}
DATASET_ORDER = ("adult", "hmda")
ENCODER_SEEDS = (0, 1, 2)
NOISE_SIGMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RELEASE_SEEDS = (0, 1, 2)
ATTACKER_SEEDS = (0, 1, 2)
TIER1_CELLS = (("adult", "income_prediction", "sex"), ("hmda", "underwriting", "race"))
TIER2_PAIRS = {
    "adult": (("income_prediction", "race"), ("employment_analysis", "race"), ("employment_analysis", "age_group"),
              ("employment_analysis", "marital_status"), ("education_assessment", "sex"),
              ("education_assessment", "race"), ("education_assessment", "income")),
    "hmda": (("underwriting", "ethnicity"), ("pricing_analysis", "race"), ("pricing_analysis", "sex"),
             ("fair_lending_audit", "race"), ("fair_lending_audit", "sex")),
}
PRIMARY_ARMS = ("A", "B", "C", "Dstar")
CONCEPT_ERASURE = {"package": "concept-erasure", "version": "0.2.4",
                   "tree_sha256": "fffac29d5914f334396f4af1f6b65ba09fcbf658fa8a013972bb50cbc1568597",
                   "tree_hash_rule": "sha256 over sorted *.py (excluding __pycache__): update('concept_erasure/'+relpath) "
                                     "then update(file bytes)",
                   "upstream_tag": "v0.2.4", "upstream_commit": "9b18b3d5c73f552798212c51d6533d649fa434cd"}
HISTORICAL_NATIVE = {"ref": "origin/main", "ref_commit": "55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827",
                     "blobs": {"adult": {"path": "results/v2_adult_ROUND4/dominant_axis_audit.json",
                                         "blob": "76f485a2147d98b1346dd74abce9c2fc3d024f4f"},
                               "hmda": {"path": "results/v2_hmda_ROUND4/dominant_axis_audit.json",
                                        "blob": "e56c885b62282483a672406988f8efe142979b9f"}}}


def d_tag(sigma: float, rs: int) -> str:
    return f"D_sigma{sigma:g}_rs{rs}"


def unit_id(dataset: str, seed: int, purpose: str, attr: str, arm: str) -> str:
    return f"{dataset}__s{seed}__{purpose}__{attr}__{arm}"


_UID = re.compile(r"^(?P<d>[a-z]+)__s(?P<s>\d+)__(?P<p>[a-z_]+?)__(?P<a>[a-z_]+?)__(?P<arm>A|B|C|D_sigma(?P<sig>[0-9.]+)"
                  r"_rs(?P<rs>\d+))$")


def parse_unit(uid: str) -> dict:
    m = _UID.match(uid)
    if not m:
        raise ValueError(f"malformed unit id {uid!r}")
    d, p, a = m["d"], m["p"], m["a"]
    if d not in DATASETS or p not in DATASETS[d]["purposes"] or a not in DATASETS[d]["purposes"][p]["disallowed_attrs"]:
        raise ValueError(f"unit id {uid!r}: dataset/purpose/attribute not in the frozen tables")
    arm = m["arm"]
    out = {"unit_id": uid, "dataset": d, "seed": int(m["s"]), "purpose": p, "attribute": a,
           "arm": arm if not arm.startswith("D") else "D", "arm_tag": arm,
           "purpose_index": DATASETS[d]["purposes"][p]["index"], "task": DATASETS[d]["purposes"][p]["task"],
           "n_task_classes": DATASETS[d]["purposes"][p]["n_task_classes"],
           "policy_set": list(DATASETS[d]["purposes"][p]["disallowed_attrs"])}
    if out["arm"] == "D":
        out.update(sigma=float(m["sig"]), release_seed=int(m["rs"]))
    out["cell"] = f"{d}__{p}__{a}"
    out["kind"] = {"A": "untreated", "B": "leace", "C": "leace", "D": "noise"}[out["arm"]]
    return out


def _arms_D():
    return [d_tag(s, r) for s in NOISE_SIGMAS for r in RELEASE_SEEDS]


def tier1_units() -> list[str]:
    out = []
    for d, p, a in TIER1_CELLS:
        for k in ENCODER_SEEDS:
            out += [unit_id(d, k, p, a, arm) for arm in ["A", "B", "C"] + _arms_D()]
    return out


def tier2_units() -> dict:
    """Fixed order: E1 (A, B) -> E2 (C) -> E3 (D grid); within each: dataset -> pair -> encoder seed -> arm."""
    e1, e2, e3 = [], [], []
    for d in DATASET_ORDER:
        for p, a in TIER2_PAIRS[d]:
            for k in ENCODER_SEEDS:
                e1 += [unit_id(d, k, p, a, "A"), unit_id(d, k, p, a, "B")]
                e2.append(unit_id(d, k, p, a, "C"))
                e3 += [unit_id(d, k, p, a, arm) for arm in _arms_D()]
    return {"E1": e1, "E2": e2, "E3": e3}


def all_registered() -> list[str]:
    t2 = tier2_units()
    return tier1_units() + t2["E1"] + t2["E2"] + t2["E3"]


def tier_of(uid: str) -> str:
    if uid in set(tier1_units()):
        return "T1"
    for k, v in tier2_units().items():
        if uid in set(v):
            return f"T2-{k}"
    raise ValueError(f"{uid} is not registered")


def primary_family() -> list[dict]:
    fam = []
    for d, p, a in TIER1_CELLS:
        cell = f"{d}__{p}__{a}"
        fam.append({"id": f"P-{d}-G1-A", "dataset": d, "cell": cell, "arm": "A", "statistic": "G1_r2",
                    "bar": 0.05, "test": "UCB < tau -> ESTABLISHED_BELOW; LCB > tau -> ESTABLISHED_ABOVE"})
        for m in PRIMARY_ARMS:
            fam.append({"id": f"P-{d}-Rrep-{m}", "dataset": d, "cell": cell, "arm": m,
                        "statistic": "C_rep_NL_macro_auc", "bar": 0.55,
                        "test": "UCB < 0.55 -> ESTABLISHED_BELOW; LCB > 0.55 -> ESTABLISHED_ABOVE"})
        for m in PRIMARY_ARMS:
            fam.append({"id": f"P-{d}-Rplus-{m}", "dataset": d, "cell": cell, "arm": m,
                        "statistic": "C_rep_plus_clean_out_NL_macro_auc", "bar": 0.55,
                        "test": "UCB < 0.55 -> ESTABLISHED_BELOW; LCB > 0.55 -> ESTABLISHED_ABOVE"})
        for m in PRIMARY_ARMS[1:]:
            fam.append({"id": f"P-{d}-U2NI-{m}", "dataset": d, "cell": cell, "arm": m,
                        "statistic": "U2_accuracy_diff_vs_A", "bar": -0.01,
                        "test": "LCB >= -0.01 -> NONINFERIOR; UCB < -0.01 -> INFERIOR; else UNRESOLVED"})
    return fam


ATT = {
    "L": {"C": LR_C, "max_iter": 5000, "solver": "lbfgs", "scaler": "standard", "select_metric": "attacker_val_log_loss",
          "description": "StandardScaler (fit on attacker_fit) + multinomial LogisticRegression"},
    "GBT": {"configs": GBT_CONFIGS, "max_iter": 500, "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 10, "scoring": "loss", "random_state": 0, "select_metric": "attacker_val_log_loss",
            "description": "HistGradientBoostingClassifier; early stopping on an internal split of attacker_fit"},
    "MLP": {"hidden_layer_sizes": [[64, 64], [128, 128], [256, 256]], "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-3, 3e-3], "max_iter": 200, "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 10, "solver": "adam", "batch_size": "auto", "random_state": 0, "scaler": "standard",
            "select_metric": "attacker_val_log_loss", "description": "StandardScaler + MLPClassifier, 18 configs"},
}

BENCH_EFFECTIVE: dict = {
    "schema": "stored_model_eval.bench_effective_protocol/v1",
    "description": "PCRL matched removal benchmark (Round-4 Adult + HMDA encoders): effective executed settings",
    "branch": BENCH_BRANCH,
    "datasets": copy.deepcopy(DATASETS),
    "panel": {
        "encoder_seeds": list(ENCODER_SEEDS),
        "tier1_cells": [{"dataset": d, "purpose": p, "attribute": a} for d, p, a in TIER1_CELLS],
        "tier1_arms": ["A", "B", "C", "D"],
        "tier2_pairs": {d: [[p, a] for p, a in v] for d, v in TIER2_PAIRS.items()},
        "tier2_order": ["E1", "E2", "E3"],
        "tier2_arms": {"E1": ["A", "B"], "E2": ["C"], "E3": ["D"]},
        "n_tier1_units": 126, "n_tier2_units": {"E1": 72, "E2": 36, "E3": 648},
        "unit_id_format": "<dataset>__s<encoder seed>__<purpose>__<attribute>__<arm>; arm in A, B, C, D_sigma<s:g>_rs<k>",
        "note": "encoder seeds, release seeds and attacker seeds are replicates over the same people, never units",
    },
    "roles": {
        "defense_fit": "defense_fit", "attacker_fit": "attacker_fit", "attacker_val": "attacker_val",
        "assessment": "assessment", "ignored_values": ["excluded", "unused", "none", ""],
        "hmda_rule": {"salt": "bench-roles-hmda-v1|", "cut_points": [0.50, 0.65],
                      "uniform": "int(sha256(salt + record_key).hexdigest()[:8], 16) / 2**32 (pilot convention)"},
        "fallback_rule": {"salt": "bench-defense-fallback-v1|", "share": 0.30},
        "adult_pilot_role_counts": {"attacker_fit": 7571, "attacker_val": 2239, "assessment": 5250},
        "description": "roles are read from INPUTS_INDEX (role 1); the runner re-derives the HMDA / fallback hash rules "
                       "and the Adult pilot counts as checks (reported by dry-run)",
    },
    "support": {"min_attacker_fit": 100, "min_attacker_val": 30, "min_assessment": 100,
                "min_supported_classes": 2, "min_defense_fit_concept": 100,
                "concept_rule": "an eraser (B or C) is fitted on the FULL declared one-hot of its concept only if every "
                                "class of every concept attribute has >= min_defense_fit_concept rows in defense_fit; "
                                "otherwise that unit is NE (nothing is dropped or pooled silently). Coordinator decision "
                                "2026-10-02",
                "pair_rule": "a pair is supported iff both classes are supported",
                "scoring_rule": "evaluation support is per role (100/30/100 on attacker_fit/attacker_val/assessment), "
                                "identical for every arm of a cell, frozen in SUPPORT_FROZEN.json before any fit",
                "task_classes": "the same 100/30/100 rule selects task classes for macro-F1/AUC; accuracy and log loss "
                                "use all assessment rows"},
    "defenses": {
        "leace": {"api": "defenses.fit_leace", "dtype": "float64", "fit_role": "defense_fit",
                  "settings": {"method": "leace", "affine": True, "constrain_cov_trace": True, "shrinkage": True,
                               "svd_tol": 0.01},
                  "settings_note": "OFFICIAL concept-erasure defaults (coordinator decision 2026-10-02); no distinct "
                                   "svd_tol=0 arm",
                  "concept_B": "onehot(target attribute), all declared classes (no pooling)",
                  "concept_C": "defenses.concat_marginal_onehots of the policy set, in disallowed_attrs order",
                  "cache_key": "(dataset, encoder seed, purpose, concept spec); fitted once, reused by every unit",
                  "alias_atol": 1e-10,
                  "alias_rule": "defenses.alias_test: C aliases B iff max|P_B - P_C| <= alias_atol and "
                                "max|mean_B - mean_C| <= alias_atol",
                  "official": CONCEPT_ERASURE},
        "noise": {"api": "defenses.noise_release", "sigmas": list(NOISE_SIGMAS), "release_seeds": list(RELEASE_SEEDS),
                  "draw_rows": "scored_roles_ascending_row_id",
                  "description": "one persistent draw per person: default_rng(release seed) over the matrix of scored-role "
                                 "rows (attacker_fit, attacker_val, assessment) in ascending row_id order"},
    },
    "native": {
        "A": {"lambda": 1e-6, "rows": "test_split", "variants": ["historical_mixed_precision", "float64"],
              "category_variant": "historical_mixed_precision", "tau": 0.05, "reproduce_atol": 1e-4,
              "historical": HISTORICAL_NATIVE,
              "label": "PCRL historical one-hot ridge R2, in-sample on the test split; compared per seed with "
                       "dominant_axis_audit.json r2_onehot"},
        "BC": {"rows": "defense_fit", "primary": "implementation_bound_holds",
               "primary_rule": "the official implementation's own specified bound: spectral norm of the whitened "
                               "residual cross-covariance on defense_fit <= svd_tol; failure -> category C1",
               "descriptive_tol_rel": 1e-6, "descriptive_tol_r2": 1e-6,
               "descriptive_rule": "exact-zero cross-covariance (crosscov_max_abs_rel_erased <= tol_rel and fit-row "
                                   "OLS R2 <= tol_r2); OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION is reported as a "
                                   "separate descriptive column, NOT C1",
               "out_of_support": "per (dataset, seed, purpose): norm of the component of (h - fit mean) outside the "
                                 "range of the defense_fit sample covariance on attacker_fit and assessment rows "
                                 "(label-free); official behaviour = identity there (transfer limitation column)",
               "label": "LEACE condition on defense_fit, finite-sample scope only"},
        "D": {"status": "NA", "label": "no native certificate (unclipped Gaussian noise; Prop 3 covers clipped only)"},
    },
    "linear_closed_form": {
        "G1": {"lambda": 1e-6, "fit_role": "attacker_fit", "score_role": "assessment", "dtype": "float64",
               "clamp": False},
        "G2": {"rho_grid": [0.0, 1e-4, 1e-2], "floor_rel": 1e-6, "select_on": "attacker_val",
               "select_metric": "heldout_r2_fitmean"},
        "rho1_heldout": {"eps_rel": 1e-6},
        "cross_cov_heldout": {"arms": ["A", "B", "C"], "score_role": "assessment",
                              "concepts": ["target", "policy"],
                              "description": "weighted Frobenius norm and max-abs of cov(released h, concept one-hot) "
                                             "on assessment rows; concept = target one-hot (B scope) and policy "
                                             "concatenated one-hot (C scope)"},
    },
    "attackers": {
        "L": ATT["L"], "GBT": ATT["GBT"], "MLP": ATT["MLP"],
        "NL": {"members": ["GBT", "MLP"], "select_metric": "attacker_val_log_loss"},
        "plus_slate": {"NL": ["GBT", "MLP", "ignore_rep", "ignore_out"], "L": ["L", "ignore_rep", "ignore_out"],
                       "ignore_rep": "the outputs_only selected model of the same family slot (NL or L)",
                       "ignore_out": "the C_rep selected model of the same family slot (NL or L)",
                       "select_metric": "attacker_val_log_loss"},
        "attacker_seeds": list(ATTACKER_SEEDS), "seed_param": "random_state",
        "deterministic_alias": ["L"],
        "retrain": "the selected recipe (family + hyper-parameters) is refit at each attacker seed; seed 0 is the "
                   "grid fit itself (random_state 0); the grid is never rerun",
        "LRT_A2": {"applies_to": "noise", "surface": "rep", "eig_floor_rel": 1e-6,
                   "floor_trace": "released_class_covariance", "cov_ddof": 1, "class_prior": "attacker_fit"},
        "LRT_A4": {"applies_to": "noise", "surface": "rep", "eig_floor_rel": 1e-6,
                   "floor_trace": "clean_class_covariance", "cov_ddof": 1, "class_prior": "attacker_fit"},
        "leace_simulation": "alias of the ordinary slate trained on erased attacker-population features (recorded)",
        "selection_role": "attacker_val",
        "log_loss_clip": 1e-12,
    },
    "surfaces": {"C_rep": "rep", "C_rep_plus_clean_out": "rep+outputs", "outputs_only": "outputs",
                 "outputs_kind": "logits",
                 "outputs_share_key": "dataset|encoder_seed|purpose|attribute",
                 "description": "outputs_only is fitted once per (dataset, encoder seed, purpose, attribute) and aliased "
                                "into every arm; the clean outputs are identical across methods"},
    "references": {"LO": {"laplace_alpha": 1.0, "fit_role": "attacker_fit"},
                   "constant": {"prior_role": "attacker_fit"}},
    "utility": {"U1": {"checkpoint_load": "weights_only", "consistency_atol": 1e-4,
                       "scope": "outside LEACE's scope: the frozen head is Linear-ReLU-Linear (nonlinear "
                                "post-processing can re-create linearly detectable signal)",
                       "description": "frozen purpose head on the transformed representation (compatibility only)"},
                "U2": {"C": LR_C, "max_iter": 5000, "fit_role": "attacker_fit", "select_role": "attacker_val",
                       "share_key": "dataset|encoder_seed|purpose|release",
                       "description": "common LR probe on the transformed representation; shared across attributes "
                                      "when the release is identical"},
                "clean_output": {"description": "the clean head's task accuracy and log loss; unchanged across methods"},
                "metrics_all_rows": ["accuracy", "log_loss"],
                "metrics_supported_task_classes": ["macro_f1", "macro_auc"]},
    "recovery_metrics": {"auc": "macro_ovr_supported", "LLR_nats": "LL0 - LL (nats per row; LL0 = attacker_fit prior)",
                         "LL_skill": "1 - LL/LL0", "brier_skill": "1 - Brier(model)/Brier(attacker_fit prior)",
                         "worst_class": "max OvR AUC over supported classes",
                         "worst_pair": "max pair AUC over supported pairs, score p_j/(p_i+p_j)", "prob_clip": 1e-12},
    "sigma_star": {"bar": 0.55, "fallback_sigma": 8.0, "attacker_seed": 0, "role": "attacker_val",
                   "statistic": "C_rep NL macro OvR AUC over supported classes",
                   "rule": "smallest sigma whose attacker_val statistic (mean over encoder seeds x release seeds, "
                           "attacker seed 0) is <= bar; none -> fallback_sigma, flagged"},
    "inference": {
        "sampling_unit": "assess_unit",
        "draw": "rng = numpy.random.default_rng(seed); replicate b: counts = rng.multinomial(n_units, full(n_units, "
                "1/n_units)); row weight = counts[unit index]; unit order = numpy.unique(assess_unit); one draw per "
                "dataset re-weights every unit, arm, encoder seed, release seed and attacker seed of that dataset",
        "quantile_method": "linear", "chunk": 500,
        "bars": [0.52, 0.55, 0.60], "tau": 0.05, "ni_margin": -0.01,
        "exploratory": {"B": 2000, "seed": 20261003, "level": 0.90},
        "primary": {"B": 20000, "seed": 20261004, "alpha_family": 0.05, "family_size": 24,
                    "adjustment": "bonferroni_simultaneous_one_sided_bounds",
                    "validity_note": "percentile bounds are asymptotic; each endpoint can err in one direction, so two "
                                     "one-sided bounds at alpha/24 bound the family-wise error by 0.05 (asymptotically)"},
        "replicate_summary": "mean over encoder seeds x release seeds x attacker seeds, computed within each bootstrap "
                             "replicate on the same resampled people",
        "worst_rule": "simultaneous per-class/pair bounds at alpha/K (K = supported classes or pairs): "
                      "LCB(max) = max_k LCB_k(alpha/K), UCB(max) = max_k UCB_k(alpha/K); point = max_k point_k",
        "max_ne_replicate_fraction": "the tail probability of the bound; more NE replicates -> NE",
    },
    "budget": {"cpu_hours": 8.0, "inference_reserve_cpu_s_per_unit": 30.0,
               "stop_rule": "before each Tier-2 unit (fixed order): stop if spent + projected(unit) + inference reserve "
                            "for all completed-or-started units > cpu_hours*3600; never skip ahead",
               "projection": "mean measured cpu of completed units of the same dataset and arm kind; calibration "
                             "estimate until one exists",
               "tier2_trigger": "every Tier-1 unit COMPLETE and hash-verified, SIGMA_STAR.json present, A native "
                                "reproduction, B/C native checks pass, U1 consistency ok, no technical failure"},
    "sanity": {"shuffle_seed": 20261005, "flag_above_auc": 0.55,
               "description": "fit/validation-only shuffled-label sanity: s permuted within attacker_fit and within "
                              "attacker_val; C_rep slate fitted and selected through the same code path; assessment never "
                              "loaded into the fit"},
    "unsupported": {
        "R12_repeated_release": "persistent contract: N collapses to 1 (A3 staged, not run)",
        "R11_adaptive_general": "LEACE arms: the ordinary slate on erased features is the defense-aware attack (alias)",
        "oracle_leace": "never used (needs target labels at inference)",
        "inhouse_leace": "pcrl.models.baselines.LEACEEraser is never used",
        "permutation_null_real": "replaced by the fit/validation-only shuffled-label sanity check",
        "U2_mlp_probe": "U2 is LR only (pilot convention)",
        "round5_7_checkpoints": "separate labelled extension; never mixed into Round-4 rows",
    },
    "threads": {"OMP_NUM_THREADS": "1"},
}

DESCRIPTIVE_LEAVES = {"description", "note", "validity_note", "draw", "pair_rule", "task_classes", "label",
                      "schema", "branch", "max_ne_replicate_fraction", "adjustment", "unit_id_format", "rule",
                      "uniform", "concept_B", "concept_C", "cache_key", "alias_rule", "official", "retrain",
                      "leace_simulation", "ignore_rep", "ignore_out", "concept_rule", "scoring_rule", "stop_rule",
                      "projection", "tier2_trigger", "replicate_summary", "worst_rule", "historical", "statistic",
                      "api", "settings_note", "primary_rule", "descriptive_rule", "out_of_support", "scope", "n_tier1_units", "n_tier2_units", "outputs_share_key", "share_key", "tier2_order",
                      "tier2_arms"}


def bench_effective_hash(eff: dict | None = None) -> str:
    return hashlib.sha256(canonical_json(BENCH_EFFECTIVE if eff is None else eff).encode()).hexdigest()


def validate_bench_effective(eff: dict | None = None) -> list[str]:
    e = BENCH_EFFECTIVE if eff is None else eff
    errs = []
    if e["datasets"] != DATASETS:
        errs.append("dataset / purpose tables differ from the frozen PCRL b96c412 tables")
    if len(tier1_units()) != 126 or e["panel"]["n_tier1_units"] != 126:
        errs.append("Tier-1 panel must have 126 units")
    t2 = tier2_units()
    if {k: len(v) for k, v in t2.items()} != e["panel"]["n_tier2_units"]:
        errs.append("Tier-2 lists differ from the registered sizes")
    s = e["support"]
    if (s["min_defense_fit_concept"], s["min_attacker_fit"], s["min_attacker_val"], s["min_assessment"]) != \
            (100, 100, 30, 100):
        errs.append("support rule must be 100 (defense_fit concept) and 100/30/100 (attacker_fit/val/assessment)")
    a = e["attackers"]
    if len(a["GBT"]["configs"]) != 20 or len(a["L"]["C"]) != 5:
        errs.append("attacker grids changed")
    if a["attacker_seeds"] != [0, 1, 2]:
        errs.append("attacker seeds must be {0,1,2}")
    I = e["inference"]
    if (I["exploratory"]["B"], I["exploratory"]["seed"], I["primary"]["B"], I["primary"]["seed"]) != \
            (2000, 20261003, 20000, 20261004):
        errs.append("bootstrap B / seeds changed")
    if I["primary"]["family_size"] != len(primary_family()) or I["primary"]["family_size"] != 24:
        errs.append("primary family must have 24 endpoints")
    if I["bars"] != [0.52, 0.55, 0.60] or I["tau"] != 0.05 or I["ni_margin"] != -0.01:
        errs.append("bars / tau / NI margin changed")
    if e["defenses"]["noise"]["sigmas"] != list(NOISE_SIGMAS) or e["defenses"]["noise"]["release_seeds"] != [0, 1, 2]:
        errs.append("noise grid changed")
    if e["sigma_star"]["bar"] != 0.55 or e["sigma_star"]["fallback_sigma"] != 8.0:
        errs.append("sigma* rule changed")
    if e["defenses"]["leace"]["official"] != CONCEPT_ERASURE:
        errs.append("official LEACE pin changed")
    return errs


def consumption_report(consumed: set, eff: dict | None = None) -> dict:
    eff = BENCH_EFFECTIVE if eff is None else eff
    leaves = leaf_paths(eff)
    unsupported = [p for p in leaves if p.startswith("unsupported")]
    descriptive = [p for p in leaves if p.split(".")[-1] in DESCRIPTIVE_LEAVES or
                   any(seg in DESCRIPTIVE_LEAVES for seg in p.split(".")[:-1])]
    rest = [p for p in leaves if p not in unsupported and p not in descriptive]
    # a list-valued leaf (e.g. GBT configs, tier lists) counts as consumed when its path was read
    return {"n_leaves": len(leaves), "consumed": sorted(p for p in rest if p in consumed),
            "descriptive": sorted(descriptive), "unsupported": sorted(eff.get("unsupported", {})),
            "not_consumed": sorted(p for p in rest if p not in consumed)}


def primary_alpha_each(eff: dict | None = None) -> float:
    p = (BENCH_EFFECTIVE if eff is None else eff)["inference"]["primary"]
    return p["alpha_family"] / p["family_size"]


def effective_json() -> str:
    return json.dumps(BENCH_EFFECTIVE, indent=1)


__all__ = ["BENCH_EFFECTIVE", "BENCH_BRANCH", "DATASETS", "TIER1_CELLS", "TIER2_PAIRS", "parse_unit", "unit_id",
           "tier1_units", "tier2_units", "all_registered", "primary_family", "bench_effective_hash",
           "validate_bench_effective", "consumption_report", "Tracked", "d_tag", "tier_of"]
