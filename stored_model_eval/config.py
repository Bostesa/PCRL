"""Protocol configuration: schema, defaults and loader.

The CLI loads a protocol JSON (e.g. notes/methodology/protocol_config.json written by the methodology
role) and deep-merges it over these defaults. Unknown keys are kept and listed (the coordinator
reconciles); keys this evaluator consumes are validated.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import SCHEMA_PROTOCOL

DEFAULT_PROTOCOL: dict = {
    "schema": SCHEMA_PROTOCOL,
    "bars": [0.52, 0.55, 0.60],
    "r2": {"tau": 0.05, "ridge_lambda": 1e-6, "fit_equals_score": True, "dtype": "float64"},
    "support": {"min_class_support": 100, "min_pair_support": 100, "macro_over": "all_declared"},
    "cca": {"eps_rel": 1e-6, "tol": 1e-10},
    "health": {"per_dim_std_min": 0.5, "effective_rank_min": 2.0, "scale_dependent": True},
    "bootstrap": {"n_boot": 2000, "alpha": 0.05, "seed": 20261002, "unit": "linkage_unit", "unit_by_dataset": {},
                  "record_key": "record_hash", "seeds_are": "refit_replicates"},
    "roles": {"names": ["defense_fit", "attacker_fit", "attacker_val", "evaluation"],
              "attacker_fit": "attacker_fit", "attacker_select": "attacker_val",
              "score": "evaluation", "disjoint_by": "unit"},
    "surfaces": ["rep", "outputs", "rep+outputs"],
    "metrics": ["macro_ovr_auc", "worst_class_auc", "worst_pair_auc", "r2_onehot_ridge", "ols_r2",
                "cca_rho2"],
    "attackers": {
        "linear": {"family": "linear", "C": [0.01, 0.1, 1.0, 10.0], "access": "A1"},
        "gbt": {"family": "nonlinear", "learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
                "max_iter": 200, "access": "A1"},
        "mlp": {"family": "nonlinear", "hidden": [[64], [128, 64]], "alpha": [1e-4, 1e-3],
                "max_iter": 200, "access": "A1"},
        "noise_lrt": {"family": "noise_lrt", "access": "A4", "requires": ["population_clean_vectors",
                                                                          "known_Sigma"]},
        "adaptive": {"family": "adaptive", "base": "gbt", "access": "A2",
                     "requires": ["mechanism_code", "public_params", "attacker_population_inputs"]},
        "repeated_release": {"family": "repeated_release", "N": [1, 4, 16], "access": "A3",
                             "valid_only_if_contract": "fresh_per_query"},
    },
    "selection": {"criterion": "log_loss", "on_role": "attacker_val"},
    "release_contract": {"noise": "none", "sigma": None},
    "guarantee_scope": {"attacker_classes": ["linear"], "surfaces": ["rep"], "metric": "r2_onehot_ridge",
                        "population": False},
    "compute": {"max_cpu_hours_pilot": 2.0},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _unknown(base: dict, over: dict, prefix="") -> list[str]:
    bad = []
    for k, v in over.items():
        if k not in base:
            bad.append(prefix + k)
        elif isinstance(v, dict) and isinstance(base[k], dict) and base[k] and k != "attackers":
            bad += _unknown(base[k], v, prefix + k + ".")
    return bad


def validate(cfg: dict) -> list[str]:
    errs = []
    bars = cfg.get("bars", [])
    if not bars or not all(isinstance(b, (int, float)) and 0.5 <= b < 1 for b in bars):
        errs.append("bars must be AUC values in [0.5, 1)")
    if cfg["support"]["min_class_support"] < 2:
        errs.append("min_class_support < 2 would let singleton classes be scored")
    if cfg["release_contract"]["noise"] not in ("none", "fresh_per_query", "persistent_token"):
        errs.append("release_contract.noise must be none|fresh_per_query|persistent_token")
    for s in cfg["surfaces"]:
        if s not in ("rep", "outputs", "rep+outputs"):
            errs.append(f"unknown surface {s}")
    if not (0 < cfg["bootstrap"]["alpha"] < 0.5):
        errs.append("bootstrap.alpha out of range")
    return errs


def translate_methodology(d: dict) -> tuple[dict, list, list]:
    """Map the methodology role's protocol_config.json (keys version/interval/support/roles/attackers as
    recipe list) onto this evaluator's schema. Returns (override dict, mapping log, unmapped top-level keys)."""
    over, log = {}, []

    def put(path, value, src):
        cur = over
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
        log.append({"from": src, "to": ".".join(path), "value": value})

    if "bars" in d:
        put(["bars"], d["bars"], "bars")
    if "primary_tau" in d:
        put(["r2", "tau"], d["primary_tau"], "primary_tau")
    iv = d.get("interval", {})
    if "B" in iv:
        put(["bootstrap", "n_boot"], iv["B"], "interval.B")
    if "alpha" in iv:
        two = "two-sided 90%" in str(iv.get("sidedness", ""))
        put(["bootstrap", "alpha"], 2 * iv["alpha"] if two else iv["alpha"],
            "interval.alpha (+sidedness: one-sided bounds = two-sided interval at 2*alpha)" if two else "interval.alpha")
    if "bootstrap_seed" in iv:
        put(["bootstrap", "seed"], iv["bootstrap_seed"], "interval.bootstrap_seed")
    if "unit" in iv:
        put(["bootstrap", "unit_by_dataset"], iv["unit"], "interval.unit")
    sp = d.get("support", {})
    if "min_class_n" in sp:
        put(["support", "min_class_support"], sp["min_class_n"], "support.min_class_n")
    if "min_pair_n" in sp:
        put(["support", "min_pair_support"], sp["min_pair_n"], "support.min_pair_n")
    if "supported" in str(sp.get("macro_over", "")):
        put(["support", "macro_over"], "supported", "support.macro_over")
    ro = d.get("roles", {})
    if "assessment" in ro:
        put(["roles", "score"], "assessment", "roles.assessment")
        put(["roles", "names"], ["defense_fit", "attacker_fit", "attacker_val", "assessment"], "roles.*")
    rec = {a.get("recipe_id"): a for a in d.get("attackers", []) if isinstance(a, dict)}
    if "R01" in rec:
        put(["attackers", "linear", "C"], rec["R01"]["hyperparameter_grid"].get("C"), "attackers.R01")
    if "R03" in rec:
        g = rec["R03"]["hyperparameter_grid"]
        put(["attackers", "gbt", "learning_rate"], g.get("learning_rate"), "attackers.R03")
        put(["attackers", "gbt", "max_leaf_nodes"], g.get("max_leaf_nodes"), "attackers.R03")
        if g.get("max_iter"):
            put(["attackers", "gbt", "max_iter"], max(g["max_iter"]), "attackers.R03 (no early stopping here)")
    if "R04" in rec:
        g = rec["R04"]["hyperparameter_grid"]
        put(["attackers", "mlp", "hidden"], g.get("hidden_layer_sizes"), "attackers.R04")
        put(["attackers", "mlp", "alpha"], g.get("alpha"), "attackers.R04")
    consumed = {"bars", "primary_tau", "interval", "support", "roles", "attackers"}
    unmapped = sorted(k for k in d if k not in consumed)
    return over, log, unmapped


def load_protocol(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_PROTOCOL)
    meta = {"source": "defaults", "sha256": None, "unknown_keys": []}
    if path:
        raw = Path(path).read_bytes()
        over = json.loads(raw)
        meta = {"source": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        if "interval" in over and "schema" not in over:  # methodology-role format
            over, log, unmapped = translate_methodology(over)
            meta.update({"format": "methodology protocol_config.json (translated)", "mapping": log,
                         "not_consumed_by_evaluator": unmapped})
        meta["unknown_keys"] = _unknown(DEFAULT_PROTOCOL, over)
        cfg = deep_merge(cfg, over)
    errs = validate(cfg)
    if errs:
        raise ValueError("protocol config invalid: " + "; ".join(errs))
    cfg["_meta"] = meta
    return cfg
