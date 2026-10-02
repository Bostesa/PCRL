"""EFFECTIVE_PROTOCOL: the exact recipe settings the CELL-A pilot runner executes (frozen before any real fit).

Source of authority: results/combined_stored_model_pilot_v1/notes/FROZEN_DESIGN.md. Every leaf key below is
either CONSUMED by the pilot code (pilot.py / recipes.py / pilot_infer.py read it through a `Tracked` view, so
consumption is checked at run time, see `consumption_report`) or DESCRIPTIVE (text only: keys named in
DESCRIPTIVE_LEAVES) or listed in the top-level "unsupported" block with a reason. `validate_effective` refuses
any other key. The hash of the canonical JSON is pinned in PILOT_LOCK_v2.json.

GBT configs: the 20 configurations below are the methodology R03 "20 random configs" drawn ONCE from the full
54-point grid lr {0.03,0.1,0.3} x leaves {15,31,63} x min_samples_leaf {20,50,100} x l2 {0,1} with
numpy.random.default_rng(20261002).choice(54, 20, replace=False) over itertools.product order, then listed
explicitly so no RNG is consulted at run time.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

BRANCH = "research/combined-stored-model-pilot-v1"

UNTREATED_UNITS = (
    "income_prediction__race", "income_prediction__sex",
    "employment_analysis__race", "employment_analysis__age_group", "employment_analysis__marital_status",
    "education_assessment__sex", "education_assessment__race", "education_assessment__income",
)
NOISE_SIGMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NOISE_SEEDS = (0, 1, 2)
NOISE_UNITS = tuple(f"income_prediction__sex__p0_sigma{s:g}_seed{k}" for s in NOISE_SIGMAS for k in NOISE_SEEDS)
REGISTERED_UNITS = UNTREATED_UNITS + NOISE_UNITS
OUTPUTS_REUSE_SOURCE = "income_prediction__sex"
PURPOSES = {"income_prediction": {"index": 0, "task": "income", "n_task_classes": 2},
            "employment_analysis": {"index": 1, "task": "occupation_group", "n_task_classes": 6},
            "education_assessment": {"index": 2, "task": "education_level", "n_task_classes": 4}}

PRIMARY_FAMILY = tuple(
    [{"id": f"P1-{u}", "unit": u, "statistic": "G1_r2", "bar": 0.05, "bar_name": "tau",
      "established_above_rule": "one-sided lower bound > 0.05 (fails to generalise)",
      "established_below_rule": "one-sided upper bound <= 0.05 (generalises)"} for u in UNTREATED_UNITS]
    + [{"id": f"P2-{u}", "unit": u, "statistic": "rep__NL__macro_auc", "bar": 0.55, "bar_name": "AUC bar",
        "established_above_rule": "one-sided lower bound > 0.55 (recovery outside scope established)",
        "established_below_rule": "one-sided upper bound < 0.55 (below bar established)"}
       for u in UNTREATED_UNITS])

GBT_CONFIGS = [
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 0.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 0.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 0.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 0.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"learning_rate": 0.3, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 1.0},
]

LR_C = [0.01, 0.1, 1.0, 10.0, 100.0]  # Addendum D1 #1: locked preparation grid

EFFECTIVE_PROTOCOL: dict = {
    "schema": "stored_model_eval.effective_protocol/v1",
    "description": "CELL-A stored-model pilot (PCRL Round-4 Adult seed 0): effective executed recipe settings",
    "branch": BRANCH,
    "units": {"registered": list(REGISTERED_UNITS), "untreated": list(UNTREATED_UNITS),
              "noise": list(NOISE_UNITS), "outputs_reuse_source": OUTPUTS_REUSE_SOURCE,
              "noise_sigmas": list(NOISE_SIGMAS), "noise_seeds": list(NOISE_SEEDS),
              "note": "release seeds are release draws over the same people, not populations or encoder seeds"},
    "purposes": copy.deepcopy(PURPOSES),
    "roles": {"attacker_fit": "attacker_fit", "attacker_val": "attacker_val", "assessment": "assessment"},
    "support": {"min_attacker_fit": 100, "min_attacker_val": 30, "min_assessment": 100,
                "min_supported_classes": 2,
                "pair_rule": "a pair is supported iff both classes are supported",
                "task_classes": "the same 100/30/100 rule selects y_task classes for macro-F1 / per-class utility / utility "
                                "macro AUC; accuracy and log-loss use all assessment rows (Addendum D1 9f)",
                "description": "frozen from counts in all three roles before any fit; NE (not estimable) is distinct "
                               "from UNRESOLVED"},
    "linear_closed_form": {
        "N0": {"lambda": 1e-6, "rows": "all", "variants": ["historical_mixed_precision", "float64"],
               "units": "untreated", "category_variant": "historical_mixed_precision", "tau": 0.05,
               "label": "historical native check: in-sample on the PCRL test split",
               "noise_units": "unverified",
               "description": "one-hot pooled ridge R2 on the unnormalised centred Gram, fit = score on all test "
                              "rows, clamped at 0 as historically (raw also stored). historical_mixed_precision = "
                              "float32 centring and Gram, float64 one-hot, cross-product, solve and R2 "
                              "(pcrl/purposes/verification.py:89-103 via eval_round4_dominant_axis.py:158-172); "
                              "float64 = everything in float64. Noise units: unverified (Addendum D1 #7)"},
        "N1": {"lambda": 1e-6, "rows": "assessment", "variants": ["historical_mixed_precision", "float64"],
               "label": "within-assessment native-form statistic (descriptive; not the historical check)",
               "description": "same estimator, fit = score on assessment rows; descriptive only"},
        "G1": {"lambda": 1e-6, "fit_role": "attacker_fit", "score_role": "assessment", "dtype": "float64",
               "ss_tot_center": "attacker_fit_means", "clamp": False},
        "G2": {"rho_grid": [0.0, 1e-4, 1e-2], "floor_rel": 1e-6, "select_on": "attacker_val",
               "select_metric": "heldout_r2_fitmean",
               "description": "R02 relative-ridge least squares (scale-invariant); secondary"},
        "rho1_heldout": {"eps_rel": 1e-6,
                         "description": "canonical directions on attacker_fit; per-row RHO_u = H b and RHO_v = a[y] "
                                        "saved on assessment; squared correlation re-weighted in the bootstrap"},
        "pure_metric": {"predictor": "G1",
                        "description": "the G1 one-hot least-squares predictions scored as macro OvR AUC "
                                       "(supported classes), predicted column k = score for class k"},
    },
    "attackers": {
        "L": {"C": LR_C, "max_iter": 5000, "solver": "lbfgs", "scaler": "standard",
              "select_metric": "attacker_val_log_loss",
              "description": "StandardScaler (fit on attacker_fit) + multinomial LogisticRegression"},
        "GBT": {"configs": GBT_CONFIGS, "max_iter": 500, "early_stopping": True, "validation_fraction": 0.1,
                "n_iter_no_change": 10, "scoring": "loss", "random_state": 0,
                "select_metric": "attacker_val_log_loss",
                "description": "HistGradientBoostingClassifier; early stopping on an internal split of attacker_fit"},
        "MLP": {"hidden_layer_sizes": [[64, 64], [128, 128], [256, 256]], "alpha": [1e-5, 1e-4, 1e-3],
                "learning_rate_init": [1e-3, 3e-3], "max_iter": 200, "early_stopping": True,
                "validation_fraction": 0.1, "n_iter_no_change": 10, "solver": "adam", "batch_size": "auto",
                "random_state": 0, "scaler": "standard", "select_metric": "attacker_val_log_loss",
                "description": "StandardScaler + MLPClassifier, full 18-config grid"},
        "NL": {"members": ["GBT", "MLP"], "select_metric": "attacker_val_log_loss",
               "description": "one NL-selected predictor per unit/surface; assessment never selects"},
        "LRT_A2": {"applies_to": "noise", "surface": "rep", "eig_floor_rel": 1e-6,
                   "floor_trace": "released_class_covariance", "cov_ddof": 1, "class_prior": "attacker_fit",
                   "description": "defense-informed Gaussian class-conditional LRT (R10, access A2): class means and "
                                  "covariances from RELEASED attacker_fit rows, covariance minus sigma^2 I, eigen-"
                                  "floored at eig_floor_rel * trace / d, predictive covariance = floored + sigma^2 I; "
                                  "no clean vectors"},
        "LRT_A4": {"applies_to": "noise", "surface": "rep", "eig_floor_rel": 1e-6,
                   "floor_trace": "clean_class_covariance", "cov_ddof": 1, "class_prior": "attacker_fit",
                   "description": "white-box population LRT (R09, access A4; labelled stress test): class Gaussians "
                                  "from CLEAN attacker_fit representations plus known Sigma = sigma^2 I"},
        "A3_repeated_query": {"status": "STAGED_NOT_RUN",
                              "description": "under the frozen contract (persistent=True, release_count one) N "
                                             "collapses to 1; a fresh-query contract is a different release "
                                             "interface needing a separate amendment"},
        "selection_role": "attacker_val",
        "log_loss_clip": 1e-12,
    },
    "surfaces": {"untreated": ["rep", "outputs", "rep+outputs"], "noise": ["rep", "rep+outputs"],
                 "noise_outputs": "reused", "outputs_kind": "logits",
                 "description": "noise-unit outputs surface = clean model output, identical across arms; its "
                                "attackers/predictions are REUSED from outputs_reuse_source and marked reused"},
    "references": {"LO": {"laplace_alpha": 1.0, "fit_role": "attacker_fit",
                          "description": "label-only reference: frequency table P(s | y_task) on attacker_fit"}},
    "utility": {"U1": {"untreated": "stored_clean_logits", "noise": "frozen_head_on_release",
                       "checkpoint_load": "weights_only", "consistency_atol": 1e-4,
                       "description": "the unit purpose's frozen head; no fitting"},
                "U2": {"C": LR_C, "max_iter": 5000, "fit_role": "attacker_fit", "select_role": "attacker_val",
                       "description": "independent LR utility probe on the released representation (LR only)"},
                "metrics_all_rows": ["accuracy", "log_loss"],
                "metrics_supported_task_classes": ["macro_f1", "macro_auc"],
                "normalised_lift": {"min_clean_lift": 0.03, "constant_predictor": "attacker_fit_majority_task_class",
                                    "description": "normalised lift (acc - acc_const)/(acc_clean - acc_const) is "
                                                   "reported only when the clean lift point is >= min_clean_lift; "
                                                   "otherwise flagged"},
                "near_ceiling_purposes": ["employment_analysis", "education_assessment"],
                "paired_reference": "untreated unit of the same purpose (income_prediction__sex for noise units)"},
    "recovery_metrics": {"auc": "macro_ovr_supported", "LLR_nats": "LL0 - LL (nats per row; LL0 = attacker_fit prior)",
                         "LL_skill": "1 - LL/LL0",
                         "brier_skill": "1 - Brier(model)/Brier(attacker_fit prior)",
                         "worst_class": "max OvR AUC over supported classes",
                         "worst_pair": "max pair AUC over supported pairs, score p_j/(p_i+p_j)",
                         "prob_clip": 1e-12},
    "inference": {
        "sampling_unit": "assess_unit",
        "draw": "rng = numpy.random.default_rng(seed); replicate b: counts = rng.multinomial(n_units, "
                "full(n_units, 1/n_units)); row weight = counts[unit index]; unit order = numpy.unique(assess_unit)",
        "quantile_method": "linear",
        "chunk": 500,
        "bars": [0.52, 0.55, 0.60],
        "tau": 0.05,
        "tau_sensitivity": [0.01, 0.02, 0.05, 0.10],
        "exploratory": {"B": 2000, "seed": 20261002, "level": 0.90},
        "primary": {"B": 20000, "seed": 20261003, "alpha_family": 0.05, "family_size": 16,
                    "adjustment": "bonferroni_simultaneous_one_sided_bounds",
                    "validity_note": "percentile-bootstrap bounds are asymptotic, not exact; each endpoint can err in "
                                     "only one direction, so two one-sided bounds at alpha/16 bound the family-wise "
                                     "error by 0.05 (asymptotically)"},
        "seed_aggregation": "mean over release seeds within each bootstrap replicate (same resampled units)",
        "max_ne_replicate_fraction": "alpha of the bound (tail) - more NE replicates make the interval NE",
    },
    "unsupported": {
        "R11_adaptive_general": "needs stored fitted defense parameters for a learned defense; the noise defense "
                                "is closed-form and covered by LRT_A2 (closed form) - not run",
        "R12_repeated_release": "STAGED: persistent contract gives N = 1 (see attackers.A3_repeated_query)",
        "R13_knows_basis": "subspace channel only; not in CELL-A",
        "R14_coalition": "multi-purpose scenario not in this pilot",
        "R15_insider": "not run in the pilot (stress test listed for completeness)",
        "R01_C_100": "FROZEN_DESIGN fixes L to C in {0.01,0.1,1,10}; methodology value 100 not executed",
        "R05_logistic_variant": "label-only reference uses the frequency table only (FROZEN_DESIGN)",
        "permutation_null": "dropped for real units (200 refits per unit exceed the allowance; Addendum D1 9c); "
                            "synthetic null/positive controls run through the same CLI path instead",
        "per_cell_real_controls": "dropped (Addendum D1 9d); synthetic controls replace them",
        "refit_seed_replicates": "single refit seed (random_state 0); refit variance not estimated (Addendum D1 9b)",
        "U2_mlp_probe": "dropped (Addendum D1 9g); U2 is LR only",
        "R11_projection_refit": "deferred, not authorised (Addendum D1 9a)",
        "stratified_role_split": "roles are the frozen pilot-roles-v1 hash split (not stratified by s,y)",
    },
    "threads": {"OMP_NUM_THREADS": "1"},
}

DESCRIPTIVE_LEAVES = {"description", "note", "validity_note", "draw", "pair_rule", "task_classes", "label",
                      "paired_reference", "schema", "branch", "max_ne_replicate_fraction", "adjustment"}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def effective_hash(eff: dict | None = None) -> str:
    return hashlib.sha256(canonical_json(EFFECTIVE_PROTOCOL if eff is None else eff).encode()).hexdigest()


def leaf_paths(d, prefix="") -> list[str]:
    out = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, Mapping) and v and prefix + k != "unsupported":
            out += leaf_paths(v, p + ".")
        else:
            out.append(p)
    return out


class Tracked(Mapping):
    """Read-only view of a nested dict that records every key path read (for the consumption check)."""

    def __init__(self, d: dict, log: set | None = None, prefix: str = ""):
        self._d, self._log, self._prefix = d, (set() if log is None else log), prefix

    def __getitem__(self, k):
        v = self._d[k]
        p = self._prefix + str(k)
        if isinstance(v, Mapping):
            return Tracked(v, self._log, p + ".")
        self._log.add(p)
        return v

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    @property
    def log(self) -> set:
        return self._log

    def plain(self) -> dict:
        return copy.deepcopy(self._d)


def consumption_report(consumed: set, eff: dict | None = None) -> dict:
    eff = EFFECTIVE_PROTOCOL if eff is None else eff
    leaves = leaf_paths(eff)
    unsupported = [p for p in leaves if p.startswith("unsupported")]
    descriptive = [p for p in leaves if p.split(".")[-1] in DESCRIPTIVE_LEAVES]
    rest = [p for p in leaves if p not in unsupported and p not in descriptive]
    not_consumed = sorted(p for p in rest if p not in consumed)
    return {"n_leaves": len(leaves), "consumed": sorted(p for p in rest if p in consumed),
            "descriptive": sorted(descriptive), "unsupported": sorted(eff.get("unsupported", {})),
            "not_consumed": not_consumed}


def validate_effective(eff: dict | None = None) -> list[str]:
    eff = EFFECTIVE_PROTOCOL if eff is None else eff
    errs = []
    if list(eff["units"]["registered"]) != list(REGISTERED_UNITS) or len(REGISTERED_UNITS) != 26:
        errs.append("registered units differ from the frozen 26-unit panel")
    if len(eff["attackers"]["GBT"]["configs"]) != 20:
        errs.append("GBT must list exactly 20 frozen configs")
    for c in eff["attackers"]["GBT"]["configs"]:
        if set(c) != {"learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}:
            errs.append(f"GBT config has unsupported keys {sorted(c)}")
    s = eff["support"]
    if (s["min_attacker_fit"], s["min_attacker_val"], s["min_assessment"]) != (100, 30, 100):
        errs.append("support rule must be 100/30/100")
    inf = eff["inference"]
    if inf["primary"]["family_size"] != len(PRIMARY_FAMILY):
        errs.append("primary family size mismatch")
    if inf["bars"] != [0.52, 0.55, 0.60] or inf["tau"] != 0.05:
        errs.append("bars/tau changed")
    return errs


def primary_alpha_each(eff: dict | None = None) -> float:
    p = (EFFECTIVE_PROTOCOL if eff is None else eff)["inference"]["primary"]
    return p["alpha_family"] / p["family_size"]


def unit_info(unit_id: str) -> dict:
    """Parse a registered unit id: purpose, attribute, and (noise units) sigma and release seed."""
    parts = unit_id.split("__")
    purpose, attr = parts[0], parts[1]
    info = {"unit_id": unit_id, "purpose": purpose, "attribute": attr, "kind": "untreated",
            "purpose_index": PURPOSES[purpose]["index"], "task": PURPOSES[purpose]["task"]}
    if len(parts) == 3:
        tag = parts[2]  # p0_sigma0.25_seed0
        p, sig, sd = tag.split("_")
        info.update(kind="noise", sigma=float(sig[len("sigma"):]), release_seed=int(sd[len("seed"):]),
                    release_tag=tag)
    return info
