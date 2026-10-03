"""Build EXECUTION_LOCK.json and PRIMARY_FAMILY.csv (before any new fit)."""
import csv, hashlib, json, sys, time
from pathlib import Path
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar import lock as LK
from oar.family import (FAMILY, FAMILY_SIZE, ALPHA_FAMILY, ALPHA_EACH, B_PRIMARY, SEED_PRIMARY, B_EXPLORATORY,
                        SEED_EXPLORATORY, LEVEL_EXPLORATORY, COMPETITIVE_CONJUNCTION)
from oar import fare_official as FO
from oar.study import SIGMA_STAR, CC_ALPHAS, CERT_SALT, CERT_SHARE, HEAD_SALT, HEAD_VAL_SHARE, CELLS
from stored_model_eval.bench_effective import BENCH_EFFECTIVE, bench_effective_hash
PKG = WT / "results/combined_output_aware_removal_v1"
prop = json.loads((PKG / "notes/fare/FARE_GRID_PROPOSAL.json").read_text())
roles = json.loads((PKG / "ROLES_AND_SUPPORT.json").read_text())
assert len(FAMILY) == FAMILY_SIZE == 12
with open(PKG / "PRIMARY_FAMILY.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "cell", "question", "delta", "margin", "kind", "pass_rule"])
    for r in FAMILY:
        w.writerow(list(r) + [f"simultaneous one-sided lower bound (alpha={ALPHA_EACH:.6g}) of delta > {r[4]}"])
assert sum(1 for _ in open(PKG / "PRIMARY_FAMILY.csv")) - 1 == FAMILY_SIZE
grid = [{k: g[k] for k in ("id", "name", "range", "max_leaf_nodes", "min_samples_leaf", "gamma", "criterion")} for g in prop["grid"]]
lock = {
    "schema": "oar_execution_lock/v1",
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "LOCKED before any new defense/head/attacker fit (development data; not a confirmation)",
    "base_commit": "70f978ffc0afff55ecdf49cf1a74908cc3db5f49",
    "worktree_head_at_build_informational": LK.git_head(),
    "code_files": LK.code_files(), "dependencies": LK.deps(), "inputs": LK.inputs_state(),
    "roles_and_support_sha256": hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest(),
    "roles_rule": {"exposure": "test records whose canon_key occurs in the encoder-training split leave every scored role",
                   "cert": {"salt": CERT_SALT, "share": CERT_SHARE, "from": "attacker_fit groups after exposure removal"},
                   "head_holdout": {"salt": HEAD_SALT, "share": HEAD_VAL_SHARE, "from": "defense_fit"},
                   "class_vocabulary": {d: {"sensitive_K": c["K_s"], "task_K": c["K_t"], "attr": c["attr"], "task": c["task"]} for d, c in CELLS.items()},
                   "support": {"thresholds": {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}, "pooling": "never"}},
    "benchmark_effective_protocol_sha256 (attackers, U2, references reused unchanged)": bench_effective_hash(BENCH_EFFECTIVE),
    "attackers": {"base": "benchmark slate (L 5, GBT 20, MLP 18; NL = GBT vs MLP on attacker_val log loss; refit seeds 0,1,2; plus-slate ignore-rep / ignore-outputs)",
                  "defense_aware": {"finite_surfaces": "cell-conditional P(s|cell), Dirichlet alpha toward attacker_fit prior", "alphas": list(CC_ALPHAS), "selection": "attacker_val log loss"},
                  "controls": {"null_seed": 20261012, "null_flag": 0.55, "planted": "append onehot(s with 20% random replacement)", "planted_detect": 0.75, "scope": "seed 0, one per new interface type, attacker_fit/val only"}},
    "noise": {"sigma_star": SIGMA_STAR, "release_seeds": [0, 1, 2], "draws": "identical to the benchmark (original scored roles, ascending row_id)"},
    "fare": {"official": {"repo": prop["official"]["repo"], "commit": prop["official"]["commit"],
                          "tree_sha256": FO.official_tree_sha256(), "sktree_tree_sha256": prop["official"]["sktree_tree_sha256"],
                          "environment": "~/PCRL_eval_cache_private/oar_v1/env_fare (notes/fare/FARE_ENV_RECIPE.md)",
                          "adaptations": "notes/fare/FARE_METHOD_NOTES.md + agent report (buffer fix, pickling, multi-group cert budget, cell ids, Lemma 5.1 counts, group recoding, seed=43+s)"},
             "grid": grid, "seed_base": prop["seed_base"], "sensitive_for_fit": prop["sensitive_for_fit"],
             "certificate": prop["certificate"], "secondary_certificate_groups": {"hmda": [0, 1, 2]},
             "supported_classes": prop["supported_classes"],
             "selection_rule": "admissible: U2 val acc >= untreated U2 val acc - 0.01; min val NL(as0) macro AUC on FARE features; ties lower id; none -> no admissible nominee, fallback = max val acc (descriptive, cannot pass conjunction)",
             "zero_fairness_control": "same k, n_min, criterion, gamma = 0, at the nominee's budget",
             "alias_rule": "identical cells to an earlier config -> alias, counted, not refitted"},
    "heads": {"family": "benchmark U2 LR grid", "fit": "defense_fit minus oar-head-v1 holdout", "select": "holdout log loss",
              "runtime_inputs": "protected features only"},
    "primary_family": {"size": FAMILY_SIZE, "rows": [list(r) for r in FAMILY], "alpha_family": ALPHA_FAMILY,
                       "alpha_each": ALPHA_EACH, "B": B_PRIMARY, "seed": SEED_PRIMARY, "quantile": "linear",
                       "bootstrap_unit": "assessment group (canon_key unit), predictors fixed",
                       "seed_aggregation": "attacker seeds -> release seeds -> encoder seeds (equal weights)",
                       "missing_rows": "UNRESOLVED, counted in the 12", "competitive_conjunction": COMPETITIVE_CONJUNCTION,
                       "empty_frontier_rule": "no admissible FARE nominee in any encoder seed of a cell -> fallback reported; P2-P6 computed on it but the conjunction cannot pass"},
    "exploratory": {"B": B_EXPLORATORY, "seed": SEED_EXPLORATORY, "level": LEVEL_EXPLORATORY},
    "runtime": {"cpu_hours_ceiling": 12.0, "reserve_cpu_s": 12600, "memory_gib": 6, "workers": 2, "blas_threads": 1,
                "order": "per dataset: seeds 0,1,2; references, outputs, A/B/C, noise, FARE grid, nominee, views, FZ, certificate; controls on seed 0",
                "mandatory": "all units; kernelized adversarial erasure optional and not planned"},
    "exposure_statement": "All rows are previously used development data; freezing a new assessment procedure does not make them fresh. 17 Adult / 42 HMDA training-overlapping test records removed from all scored roles.",
    "registered_predictions_sha256": hashlib.sha256((PKG / "PREDICTIONS.md").read_bytes()).hexdigest(),
    "disclosure": "for FARE planning the admission owner read aggregate race-group shares of attacker_fit and aggregate task/group frequencies of defense_fit (no features, no fits, no outcomes)",
}
out = PKG / "EXECUTION_LOCK.json"
out.write_text(json.dumps(lock, indent=1))
v = LK.verify_lock(out)
print("lock written; verify:", v)
