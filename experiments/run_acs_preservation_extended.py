"""Bounded all-release nested audit extension after the preservation core.

No representation is fitted, changed or selected here. Each saved release and
original auditor is hash checked; 120-epoch prefixes must replay the original
selected predictions exactly before any development-evaluation labels open.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.acs_transfer_data import (
    array_hash, audit_labels, load_cohort, sha_file, split_households, write_json,
)
from experiments.acs_transfer_heads import _hash_array, _state_hash, load_candidate
from experiments.acs_protection_audits import select_candidates
from experiments.acs_preservation_audits import (
    fit_extended_auditors, fit_extended_catchup, save_extended_audits,
)
from experiments.run_acs_protection import ROOT, AUDITS, frozen_digest, now, read, score_and_save
from experiments.run_acs_transfer import subset_indices


STATIC_IDS = ("logistic", "hist_gb_20", "hist_gb_5")
FRESH_IDS = ("mlp_0", "mlp_1")


def release_output_hashes(freeze, original):
    """Accept the separately frozen, single-release PCA16 historical schema."""
    if original == "PCA16":
        assert freeze["release"] == "PCA16" and freeze["dimension"] == 16
        assert freeze["component_indices"] == list(range(16))
        assert freeze["representation_fitted"] is False
        return freeze["output_hashes"]
    return freeze["output_hashes"][original]


def release_sources(out, cfg, seed):
    """Frozen order: all eight new arms, both beta0 arms, PCA16, rich banks."""
    sources = []
    for beta in ("beta_0p1", "beta_1"):
        for arm in cfg["continuation_arms"]:
            sources.append((beta+"_"+arm, out/beta/f"seed_{seed}", arm))
    for key, names in (("init_reference_results", ("C_init", "D_init")),
                       ("pca16_reference_results", ("PCA16",)),
                       ("reference_results", ("B_rich_bank", "C_tree_bank"))):
        for name in names:
            sources.append((name, ROOT/cfg[key]/f"seed_{seed}", name))
    return sources


def verify_nested(original, nested, xval):
    """Historical first-budget predictions, state, schedules and scores replay."""
    for field in ("initial_state_hash", "final_state_hash", "selected_state_hash", "schedule_hash",
                  "optimizer_steps", "training_row_exposures", "selected_epoch", "validation_curve"):
        if field in original.metadata:
            assert nested.metadata[field] == original.metadata[field], "Nested120 replay changed: "+field
    for field in ("fit_hashes", "validation_hashes", "validation_scores", "fit_support", "n_classes"):
        assert nested.metadata[field] == original.metadata[field], "Nested120 recipe changed: "+field
    assert np.array_equal(nested.predict_proba(xval), original.predict_proba(xval)), "Nested120 predictions changed"
    if original.preprocessing is not None:
        assert np.array_equal(nested.preprocessing.mean, original.preprocessing.mean)
        assert np.array_equal(nested.preprocessing.scale, original.preprocessing.scale)
    return {"selected_state_hash": nested.metadata.get("selected_state_hash"),
            "validation_probability_sha256": _hash_array(nested.predict_proba(xval)),
            "exact_original_prefix": True}


def _selection(candidates):
    scores = {cid: candidate.metadata["validation_scores"] for cid, candidate in candidates.items()}
    primary = select_candidates({cid: value for cid, value in scores.items() if cid != "saved_adversary"})
    independent = select_candidates({cid: value for cid, value in scores.items() if cid not in ("catchup", "saved_adversary")})
    family = lambda cid: cid if cid in ("catchup", "saved_adversary") else candidates[cid].metadata["family"]
    families = {name: min((cid for cid in candidates if family(cid) == name),
                         key=lambda cid: (scores[cid]["log_loss"], cid))
                for name in {family(cid) for cid in candidates}}
    metadata = {**primary, "candidate_ids": list(candidates), "validation_scores": scores,
                "candidates": {cid: c.metadata for cid, c in candidates.items()},
                "independent_selection": independent, "family_selections": families,
                "saved_adversary_diagnostic_only": True,
                "scope": "finite predictive audit, no privacy guarantee; inherited fitting exposure remains separate"}
    return primary, independent, families, metadata


def run_extended_seed(out, cfg, seed):
    from experiments.run_acs_preservation import verify_freeze, execution_amendments
    verify_freeze(out)
    for beta in ("beta_0p1", "beta_1"):
        for s in cfg["seeds"]:
            assert (out/beta/f"seed_{s}"/"metrics.json").exists(), "Complete entire core before optional audits"
    started = time.perf_counter()
    directory = out/"extended"/f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    pcfg = read(ROOT/cfg["parent_results"]/"config.json")
    support = read(ROOT/cfg["parent_results"]/"schema_support.json")
    assert sha_file(ROOT/pcfg["raw_path"]) == support["raw_sha256"]
    frame, cohort = load_cohort(ROOT/pcfg["raw_path"], pcfg["sample_cap"], pcfg["sample_seed"])
    pools = split_households(frame, seed)
    yf = audit_labels(frame.iloc[pools["attacker_fit"]])
    yv = audit_labels(frame.iloc[pools["attacker_validation"]])
    indices = {target: subset_indices(yf[target], cfg["attacker_budget"], 1240000+100*seed+j)
               for j, target in enumerate(AUDITS)}
    sources = release_sources(out, cfg, seed)
    manifests, selections, used = {}, {}, {}

    def checked(parent, path):
        if parent not in manifests:
            manifests[parent] = {str(ROOT/v["path"]): v["sha256"] for v in read(parent/"local_artifacts.json")}
            selections[parent] = read(parent/"selection_before_test.json")
            with np.load(parent/"split_rows.npz") as rows:
                for pool, ix in pools.items():
                    assert np.array_equal(rows[pool], frame.iloc[ix]._raw_row.to_numpy()), "Household identity changed"
        digest = sha_file(path)
        assert manifests[parent].get(str(path)) == digest, "Missing/changed historical object: "+str(path)
        used[str(path.relative_to(ROOT))] = digest
        return path

    def candidate(parent, oldkey, cid):
        path = parent/"fitted"/oldkey/cid
        for file in path.iterdir():
            if file.suffix in (".npz", ".pt", ".joblib"):
                checked(parent, file)
        fitted = load_candidate(path)
        assert fitted.metadata == selections[parent]["fitting_records"][oldkey]["candidates"][cid]
        return fitted

    releases, original_release_paths = {}, {}
    for canonical, parent, original in sources:
        path = checked(parent, parent/f"release_{original}.npz")
        original_release_paths[canonical] = path
        with np.load(path) as arrays:
            releases[canonical] = {pool: arrays[pool].copy() for pool in arrays.files if pool != "test"}
        freeze = read(parent/"release_freeze.json")
        expected_hashes = release_output_hashes(freeze, original)
        for pool, values in releases[canonical].items():
            assert array_hash(values) == expected_hashes[pool]
            values.setflags(write=False)
    before = frozen_digest(releases)
    write_json(directory/"release_freeze.json", {
        "created_utc": now(), "output_hashes": before, "execution_amendments_sha256": execution_amendments(out),
        "source_files_sha256": copy.deepcopy(used), "mapper_fitting": False,
        "release_order": [name for name, _, _ in sources],
        "evaluation_status": cfg["evaluation_status"]})
    fitted = {budget: {} for budget in (120, 360)}
    records = {str(b): {key: {} for key in ("head_selections", "independent_selections", "family_selections",
                                           "auroc_selections", "fitting_records")} for b in fitted}
    parity, fresh_seconds, catchup_seconds = {}, 0., 0.
    all_sources = [*sources, ("exposed", ROOT/cfg["reference_results"]/f"seed_{seed}", "exposed")]
    for canonical, parent, original in all_sources:
        print(f"extended seed {seed}: {canonical}: nested120/360 fresh and saved-start audits", flush=True)
        for j, (target, classes) in enumerate(AUDITS.items()):
            ix, valid = indices[target], np.flatnonzero(yv[target] >= 0)
            fit_labels, val_labels = yf[target][ix], yv[target][valid]
            if canonical == "exposed":
                xf, xv = np.eye(classes)[fit_labels], np.eye(classes)[val_labels]
            else:
                xf, xv = releases[canonical]["attacker_fit"][ix], releases[canonical]["attacker_validation"][valid]
            oldkey, key = f"audit/{original}/{target}", f"audit/{canonical}/{target}"
            statics = {cid: candidate(parent, oldkey, cid) for cid in STATIC_IDS}
            for cid in STATIC_IDS:
                assert statics[cid].metadata["fit_hashes"] == {"x": _hash_array(np.asarray(xf, np.float64)), "y": _hash_array(fit_labels)}
            tick = time.perf_counter()
            fresh = fit_extended_auditors(xf, fit_labels, xv, val_labels, classes, 1260000+100*seed+j,
                                          static_candidates=statics)
            fresh_seconds += time.perf_counter()-tick
            save_extended_audits(fresh, directory/"fitted"/key/"fresh")
            parity[key] = {cid: verify_nested(candidate(parent, oldkey, cid),
                            fresh["nested120"]["candidates"][cid], xv) for cid in FRESH_IDS}
            catchup = None
            if "saved_adversary" in selections[parent]["fitting_records"][oldkey]["candidates"]:
                saved = candidate(parent, oldkey, "saved_adversary")
                tick = time.perf_counter()
                catchup = fit_extended_catchup(saved.model, xf, fit_labels, xv, val_labels,
                    classes, 1300000+100*seed+j, inherited_exposure=saved.metadata["inherited_exposure"])
                catchup_seconds += time.perf_counter()-tick
                save_extended_audits(catchup, directory/"fitted"/key/"saved_start")
                parity[key]["catchup"] = verify_nested(candidate(parent, oldkey, "catchup"), catchup["nested120"], xv)
                assert np.array_equal(saved.predict_proba(xv), catchup["saved"].predict_proba(xv))
            for budget, nested in ((120, "nested120"), (360, "nested360")):
                candidates = dict(fresh[nested]["candidates"])
                if catchup:
                    candidates.update(catchup=catchup[nested], saved_adversary=catchup["saved"])
                primary, independent, families, metadata = _selection(candidates)
                metadata.update(audit_budget=budget, fresh_trajectory=fresh["metadata"][nested],
                                catchup_trajectory=catchup[nested].metadata if catchup else None)
                fitted[budget][key] = candidates
                record = records[str(budget)]
                record["head_selections"][key] = primary["selected_family"]
                record["independent_selections"][key] = independent["selected_family"]
                record["family_selections"][key] = families
                record["auroc_selections"][key] = primary["selected_auroc"]
                record["fitting_records"][key] = metadata
        write_json(directory/"fit_progress.json", {"updated_utc": now(), "completed_release": canonical,
            "completed_attributes": list(AUDITS), "elapsed_seconds": time.perf_counter()-started,
            "development_evaluation_opened": False})
    # The unchanged prior is fitted only to the original attacker-fitting labels.
    parent = ROOT/cfg["reference_results"]/f"seed_{seed}"
    for target, classes in AUDITS.items():
        key = f"audit/prior/{target}"
        old_record = selections[parent]["fitting_records"][key]
        priors = {cid: candidate(parent, key, cid) for cid in old_record["candidates"]}
        for prior in priors.values():
            assert prior.metadata["fit_label_hash"] == _hash_array(yf[target][indices[target]])
            assert prior.metadata["pseudocount_per_class"] == 1.
        for budget in fitted:
            primary, independent, families, metadata = _selection(priors)
            fitted[budget][key] = priors
            record = records[str(budget)]
            record["head_selections"][key] = primary["selected_family"]
            record["independent_selections"][key] = independent["selected_family"]
            record["family_selections"][key] = families
            record["auroc_selections"][key] = primary["selected_auroc"]
            record["fitting_records"][key] = metadata
    record = {"created_utc": now(), "evaluation_status": cfg["evaluation_status"], "budgets": records,
              "source_files_sha256": used, "release_freeze_sha256": sha_file(directory/"release_freeze.json"),
              "protocol_freeze_sha256": sha_file(out/"protocol_freeze.json"), "execution_amendments_sha256": execution_amendments(out), "nested120_exact_replay": parity,
              "audit_fit_indices": {target: {"n": len(ix), "pool_indices_sha256": array_hash(ix),
                "raw_rows_sha256": array_hash(frame.iloc[pools["attacker_fit"][ix]]._raw_row.to_numpy())}
                for target, ix in indices.items()}}
    write_json(directory/"selection_before_test.json", record)
    selection_hash = sha_file(directory/"selection_before_test.json")
    evaluation_started = now()
    print(f"extended seed {seed}: all selections frozen; opening DEVELOPMENT EVALUATION", flush=True)
    test = {}
    for canonical, path in original_release_paths.items():
        with np.load(path) as arrays:
            test[canonical] = arrays["test"].copy()
        test[canonical].setflags(write=False)
    test_frame = frame.iloc[pools["test"]]
    test_y = audit_labels(test_frame)
    predictions, raw = {}, []
    for budget in fitted:
        current_predictions = {}
        selected = records[str(budget)]
        val = score_and_save(fitted[budget], selected["head_selections"],
            {"audit": {n: a["attacker_validation"] for n, a in releases.items()}}, {"audit": yv},
            {"audit": frame.iloc[pools["attacker_validation"]].PWGTP.to_numpy(float)},
            "validation", current_predictions, cfg)
        evaluated = score_and_save(fitted[budget], selected["head_selections"], {"audit": test}, {"audit": test_y},
            {"audit": test_frame.PWGTP.to_numpy(float)}, "test", current_predictions, cfg)
        predictions.update({f"budget{budget}/"+key: value for key, value in current_predictions.items()})
        for key, candidates in fitted[budget].items():
            role, release, target = key.split("/")
            for cid, candidate in candidates.items():
                family = cid if cid in ("catchup", "saved_adversary") else candidate.metadata["family"]
                raw.append({"seed": seed, "role": role, "release": release, "parent": "E_pca", "erased": False,
                    "target": target, "candidate_id": cid, "family": family, "audit_budget": budget,
                    "selected": selected["head_selections"][key] == cid,
                    "independent_selected": selected["independent_selections"][key] == cid,
                    "selected_within_family": selected["family_selections"][key][family] == cid,
                    "auc_selected": selected["auroc_selections"][key] == cid,
                    "validation": val[key, cid]["score"], "validation_person_weighted": val[key, cid]["weighted"],
                    "test": evaluated[key, cid]["score"], "test_person_weighted": evaluated[key, cid]["weighted"],
                    "reused_reference": cid in STATIC_IDS or release == "prior",
                    "trajectory": "same prescribed 360-epoch path; nested validation selections"})
    np.savez_compressed(directory/"predictions.npz", **predictions)
    np.savez_compressed(directory/"split_rows.npz", **{pool: frame.iloc[ix]._raw_row.to_numpy() for pool, ix in pools.items()})
    integrity = {"releases_unchanged": frozen_digest(releases) == before,
                 "source_files_unchanged": all(sha_file(ROOT/path) == digest for path, digest in used.items()),
                 "selection_unchanged": sha_file(directory/"selection_before_test.json") == selection_hash,
                 "selection_created_utc": record["created_utc"], "evaluation_started_utc": evaluation_started,
                 "selection_sha256": selection_hash, "nested120_exact_replay": True,
                 "evaluation_output_hashes": {n: array_hash(x) for n, x in test.items()}}
    assert all(v for key, v in integrity.items() if key.endswith("_unchanged"))
    local = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha_file(path),
              "availability": "local only; fixed reconstruction instructions in study report"}
             for path in directory.rglob("*") if path.is_file() and path.suffix in (".pt", ".npz", ".joblib")]
    write_json(directory/"local_artifacts.json", local)
    runtime = {"total_seconds": time.perf_counter()-started, "independent_audit_seconds": fresh_seconds,
               "catchup_seconds": catchup_seconds, "representation_training_seconds": 0.}
    result = {"seed": seed, "evaluation_status": cfg["evaluation_status"], "cohort": cohort,
              "raw_metrics": raw, "integrity": integrity, "runtime": runtime,
              "complete_release_order": [name for name, _, _ in all_sources],
              "completed_budgets": [120, 360], "all_13_releases_and_exposed_controls_complete": True}
    write_json(directory/"metrics.json", result)
    print(json.dumps({"extended_seed": seed, "runtime": runtime, "integrity": integrity}), flush=True)
    return result


def estimate_remaining_seconds(out, cfg, remaining_seeds):
    """Conservative cost projection, using core timings without viewing outcomes."""
    extended = [read(out/"extended"/f"seed_{seed}"/"metrics.json")["runtime"]["total_seconds"]
                for seed in cfg["seeds"] if (out/"extended"/f"seed_{seed}"/"metrics.json").exists()]
    if extended:
        return max(extended)*remaining_seeds*1.25
    runtimes = [read(out/beta/f"seed_{seed}"/"metrics.json")["runtime"]
                for beta in ("beta_0p1", "beta_1") for seed in cfg["seeds"]]
    # Core has four releases per unit; extension has fourteen fresh suites and
    # ten saved-adversary suites. Counting reused tree cost again is conservative.
    per_seed = (max(r["independent_audit_seconds"] for r in runtimes)*14/4*3
                + max(r["catchup_seconds"] for r in runtimes)*10/4*3 + 15)
    return per_seed*remaining_seeds*1.25


def main():
    from experiments.run_acs_preservation import DEFAULT_OUT, verify_freeze
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    cfg = read(out/"config.json")
    verify_freeze(out)
    progress_path = out/"progress.json"
    progress = read(progress_path)
    assert progress["status"] == "core complete", "Optional extension follows the complete core"
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in cfg["seeds"]:
            if (out/"extended"/f"seed_{seed}"/"metrics.json").exists():
                assert seed in progress.get("completed_extended_seeds", []), "Missing runtime accounting"
                continue
            remaining_seeds = len(cfg["seeds"])-len(progress.get("completed_extended_seeds", []))
            estimate = estimate_remaining_seconds(out, cfg, remaining_seeds)
            elapsed = (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg["started_utc"])).total_seconds()
            if (progress["scientific_seconds"]+estimate > cfg["maximum_scientific_seconds"]
                    or elapsed+estimate > cfg["maximum_total_work_seconds"]-900):
                progress.update(extended_status="not run" if remaining_seeds == len(cfg["seeds"]) else "partial",
                    extended_reason="Conservative complete-extension projection exceeds remaining scientific or total-work ceiling",
                    extended_projected_remaining_seconds=estimate, updated_utc=now())
                write_json(progress_path, progress)
                return
            tick = time.perf_counter()
            try:
                run_extended_seed(out, cfg, seed)
            except Exception as error:
                progress["scientific_seconds"] += time.perf_counter()-tick
                progress.update(extended_status="partial operational failure", extended_failed_seed=seed,
                    extended_reason=type(error).__name__+": "+str(error), updated_utc=now())
                write_json(progress_path, progress)
                raise
            seconds = time.perf_counter()-tick
            progress["scientific_seconds"] += seconds
            progress.setdefault("completed_extended_seeds", []).append(seed)
            progress.setdefault("extended_seed_seconds", []).append(seconds)
            progress.update(extended_status="running", updated_utc=now())
            write_json(progress_path, progress)
        progress.update(extended_status="complete", updated_utc=now())
        write_json(progress_path, progress)


if __name__ == "__main__":
    main()
