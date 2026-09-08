"""Freeze validation-only choices among existing teacher-composed witnesses.

This read-only analysis never fits a model. The first phase reads only training,
release and attacker-validation evidence. It writes an immutable choice manifest
before the second phase opens development scores or predictions. Candidate
fidelity itself is replayed independently by verify_acs_restricted.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check

DEFAULT = ROOT / "results/redesign_20260908_acs_restricted_inputs_v1"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def choose_witnesses(witnesses):
    """Choice sees only validation loss, IDs and declared candidate scope."""
    groups = {}
    for row in witnesses:
        key = f"{row['teacher']}/seed_{row['seed']}/{row['target']}/budget{row['audit_budget']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, rows in sorted(groups.items()):
        scopes = {
            "independent": [r for r in rows if r["candidate_id"] not in ("catchup", "saved_adversary")],
            "catchup": [r for r in rows if r["candidate_id"] == "catchup"],
            "pooled": [r for r in rows if r["candidate_id"] != "saved_adversary"],
        }
        result[key] = {scope: min(eligible, key=lambda r: (r["validation_log_loss"], r["witness_id"]))["witness_id"] if eligible else None
                       for scope, eligible in scopes.items()}
    return result


def parameter_accounting(checkpoint, candidate_path, metadata, bank, cid):
    """Count frozen inference parameters and list each preprocessing stage."""
    import joblib
    import numpy as np
    import torch
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')['model_state']
    count = lambda prefix: sum(value.numel() for key, value in state.items() if key.startswith(prefix+'.'))
    mapper, heads, decoder = (count(prefix) for prefix in ('mapper', 'heads', 'decoder'))
    inactive = state['mapper.0.weight'][:, 16:].numel()
    assert mapper == 4176 and heads == 51 and decoder == 3168 and inactive == 2048
    assert torch.count_nonzero(state['mapper.0.weight'][:, 16:]) == 0
    family = metadata['family']
    if family == 'mlp':
        auditor = torch.load(candidate_path/'model.pt', weights_only=True, map_location='cpu')['state']
        auditor_count = sum(value.numel() for value in auditor.values())
        if 'parameter_count' in metadata:
            assert metadata['parameter_count'] == auditor_count
        auditor_shape = {'trainable_numeric_parameters': auditor_count,
                         'affine_layers': 3, 'hidden_relu_layers': 2,
                         'layer_widths': [metadata['input_dim'], 64, 32, metadata['n_classes']],
                         'output': 'softmax with full declared class schema'}
    else:
        model = joblib.load(candidate_path/'model.joblib')
        if family == 'logistic':
            auditor_shape = {'trainable_numeric_parameters': int(model.coef_.size+model.intercept_.size),
                             'affine_layers': 1, 'coefficient_shape': list(model.coef_.shape),
                             'output': 'logistic probabilities expanded to full declared class schema'}
        else:
            predictors = [tree for iteration in model._predictors for tree in iteration]
            auditor_shape = {'trainable_numeric_parameters': None,
                             'parameter_count_definition': 'tree nodes mix numeric values and discrete structure; report actual tree structure',
                             'trees': len(predictors), 'nodes': sum(len(tree.nodes) for tree in predictors),
                             'leaves': sum(int(tree.nodes['is_leaf'].sum()) for tree in predictors),
                             'maximum_tree_depth': max(int(tree.nodes['depth'].max()) for tree in predictors),
                             'baseline_prediction_scalars': int(model._baseline_prediction.size),
                             'output': 'boosted-tree probabilities expanded to full declared class schema'}
    with np.load(candidate_path/'preprocessing.npz') as arrays:
        pre = {key: arrays[key].copy() for key in ('mean', 'scale')}
    return {'teacher_input_dimension': 16, 'stored_model_input_dimension': 48,
            'counting_boundary': 'composition begins at already published teacher T; its preexisting eraser is held fixed and outside added witness parameters',
            'raw_input_coordinates_supplied': 0, 'internally_constant_zero_coordinates': 32,
            'mapper': {'stored_parameters': mapper, 'effective_parameters': mapper-inactive,
                       'inactive_raw_weights': inactive, 'affine_layers': 2, 'hidden_relu_layers': 1,
                       'layer_widths': [48, 64, 16]},
            'source_heads': {'stored_parameters': heads, 'used_in_predictive_composition': bank,
                             'affine_layers': 1 if bank else 0, 'sigmoid_output_coordinates': 3 if bank else 0},
            'decoder': {'stored_parameters': decoder, 'used_in_training_or_inference': False},
            'frozen_model_inference_parameters_stored': mapper+(heads if bank else 0),
            'frozen_model_inference_parameters_effective': mapper-inactive+(heads if bank else 0),
            'auditor': auditor_shape, 'auditor_fixed_parameters': metadata['parameters'],
            'preprocessing': {
                'teacher': 'raw frozen teacher16 coordinates; subtract original representation-fitting PCA16 mean and divide by original scales',
                'original_mean_sha256': check.array_hash(state['input_mean'].numpy()),
                'original_scale_sha256': check.array_hash(state['input_scale'].numpy()),
                'auditor': 'identity direct coordinates; no new standardizer' if cid in ('saved_adversary', 'catchup') else 'mean/std from original attacker fitting examples only',
                'auditor_mean_sha256': check.array_hash(pre['mean']), 'auditor_scale_sha256': check.array_hash(pre['scale']),
                'auditor_mean': pre['mean'].tolist(), 'auditor_scale': pre['scale'].tolist()}}


def collect_validation(out):
    """Do not open metrics, composition parity, native or development arrays."""
    cfg = check.read(out / "config.json")
    witnesses, manifests = [], {}
    complete = []
    for unit in cfg["execution_order"]:
        teacher, access, seed = unit
        if access == "F":
            continue
        directory = out / f"{teacher}_{access}" / f"seed_{seed}"
        if not (directory / "completion.json").exists():
            continue
        complete.append(unit)
        selection = check.read(directory / "selection_before_test.json")
        frozen = check.read(directory / "release_freeze.json")
        training = check.read(directory / "training/training.json")
        support = check.read(directory / 'support.json')
        assert selection["release_freeze_sha256"] == check.sha(directory / "release_freeze.json")
        assert selection['protocol_freeze_sha256'] == check.sha(out/'protocol_freeze.json')
        assert not training["raw_inputs_received"] and training["access"] == "K"
        relative = str(directory.relative_to(out))
        manifests[relative] = {p: check.sha(directory / p) for p in (
            "selection_before_test.json", "release_freeze.json", "training/training.json", "completion.json")}
        teacher_path = ROOT / cfg["selective_reference_results"] / "static" / f"seed_{seed}" / "teachers/teachers.json"
        teacher_meta = check.read(teacher_path)
        for budget in (120, 360):
            records = selection["budgets"][str(budget)]
            for key, fitted in sorted(records["fitting_records"].items()):
                role, release, target = key.split("/")
                assert role == "audit" and release in (("B",) if access == "bank" else ("C", "D"))
                for cid, metadata in sorted(fitted["candidates"].items()):
                    cpath = directory / "fitted" / key
                    if cid == "saved_adversary":
                        cpath = cpath / "saved_start/saved"
                    else:
                        cpath = cpath / ("saved_start" if cid == "catchup" else "fresh") / f"nested{budget}" / cid
                    checkpoint = directory / "training" / release / "final.pt"
                    assert metadata == check.read(cpath / "metadata.json")
                    arm = training["arms"][release]
                    witnesses.append({
                        "witness_id": f"{relative}/{key}/budget{budget}/{cid}",
                        "unit": unit, "directory": relative, "teacher": teacher, "seed": seed,
                        "source_release": release, "bank": access == "bank", "target": target,
                        "candidate_id": cid, "family": metadata["family"], "audit_budget": budget,
                        "scope": "saved_diagnostic" if cid == "saved_adversary" else ("catchup" if cid == "catchup" else "independent"),
                        "eligible": cid != "saved_adversary", "raw_input_argument_supplied": False,
                        "validation_log_loss": metadata["validation_scores"]["log_loss"],
                        "validation_scores": metadata["validation_scores"],
                        "attacker_validation_pool_raw_rows_sha256": support['attacker_validation']['raw_row_sha256'],
                        "attacker_fitting_subset_raw_rows_sha256": selection['audit_fit_indices'][target]['raw_rows_sha256'],
                        "validation_feature_label_sha256": metadata['validation_hashes'],
                        "selected_independent_on_source": records["independent_selections"][key] == cid,
                        "selected_pooled_on_source": records["head_selections"][key] == cid,
                        "selected_epoch": metadata.get("selected_epoch"),
                        "selected_epoch_zero": metadata.get("selected_epoch") == 0,
                        "model_checkpoint": str(checkpoint.relative_to(ROOT)), "model_checkpoint_sha256": check.sha(checkpoint),
                        "model_state_sha256": frozen["state"]["models"][release],
                        "auditor_directory": str(cpath.relative_to(ROOT)),
                        "auditor_files_sha256": {p.name: check.sha(p) for p in sorted(cpath.iterdir()) if p.is_file()},
                        "parameter_depth_preprocessing": parameter_accounting(checkpoint, cpath, metadata, access == 'bank', cid),
                        "teacher_manifest_sha256": check.sha(teacher_path),
                        "teacher_map_sha256": teacher_meta["maps"][teacher]["map_sha256"],
                        "composition": "frozen teacher16 -> 48-input mapper with internally zero raw block -> auditor" if access != "bank"
                                       else "frozen teacher16 -> teacher-only mapper -> three fixed native source probabilities -> auditor",
                        "exposure": {
                            "teacher_map": {"labels": "joint real SEX/RAC1P" if teacher == "E" else "paired-permuted SEX/RAC1P",
                                            "fit_pool": "representation_fit", "preexisting_frozen_map": True},
                            "source_model": training["native_source_head_training_exposure"],
                            "source_model_label_sha256": training["source_label_hashes"],
                            "real_protected_mapper_gradient": release == "D",
                            "real_protected_mapper_epochs": 80 if release == "D" else 0,
                            "observer_optimizer_steps": arm["optimizer_counts_including_common"]["adversary_optimizer_steps"],
                            "observer_row_exposures": training["common_adversary_row_exposures"] + arm["adversary_row_exposures"],
                            "observer_parameters_in_predictive_composition": cid in ("catchup", "saved_adversary"),
                            "attacker_training": {k: metadata.get(k) for k in ("fit_hashes", "optimizer_steps", "training_row_exposures", "per_row_exposures")},
                            "inherited_catchup": metadata.get("inherited_exposure"),
                            "comparison_scope": "expanded witness depth and representation-fitting label exposure; not a matched direct-teacher audit",
                        },
                    })
    return cfg, witnesses, manifests, complete


def compose(out):
    started = time.perf_counter()
    assert not (out / "COMPOSED_SELECTION.json").exists(), "Preserve existing frozen choices"
    cfg, witnesses, manifests, complete = collect_validation(out)
    choices = choose_witnesses(witnesses)
    freeze = {"created_utc": now(), "protocol_freeze_sha256": check.sha(out / "protocol_freeze.json"),
              "script_sha256": check.sha(__file__), "evaluation_status": cfg["evaluation_status"],
              "selection_inputs": "attacker-validation losses only, lexicographic witness-ID ties",
              "development_predictions_opened": False, "standalone_saved_adversary_eligible": False,
              "catchup_epoch_zero_eligible": True, "completed_eligible_units": complete,
              "complete_matrix": len(complete) == 12, "unit_manifests_sha256": manifests,
              "witnesses": witnesses, "choices": choices}
    with (out / "COMPOSED_SELECTION.json").open("x") as handle:
        json.dump(freeze, handle, indent=2, allow_nan=False)
        handle.write("\n")
    # Only now access development scores and candidate fidelity records.
    evaluation_started = now()
    evaluated, evidence = {}, {}
    for relative in manifests:
        directory = out / relative
        measured = check.read(directory / "metrics.json")
        parity = check.read(directory / "composition_parity.json")
        for row in measured["raw_metrics"]:
            if row["role"] != "audit":
                continue
            key = (relative, row["release"], row["target"], row["audit_budget"], row["candidate_id"])
            evaluated[key] = row
        for record in parity["all_existing_audit_candidates"]:
            key = (relative, record["source_release"], record["target"], record["audit_budget"], record["candidate_id"])
            evidence[key] = record
    rows = []
    for witness in witnesses:
        key = (witness["directory"], witness["source_release"], witness["target"], witness["audit_budget"], witness["candidate_id"])
        scores, parity = evaluated[key], evidence[key]
        assert parity["teacher_only_composition"] and not parity["raw_input_argument_supplied"]
        assert witness["validation_scores"] == scores["validation"]
        group = f"{witness['teacher']}/seed_{witness['seed']}/{witness['target']}/budget{witness['audit_budget']}"
        rows.append({**witness, "selected_global": {scope: wid == witness["witness_id"] for scope, wid in choices[group].items()},
                     "validation_person_weighted": scores["validation_person_weighted"],
                     "test": scores["test"], "test_person_weighted": scores["test_person_weighted"], "composition_parity": parity["splits"]})
    result = {"created_utc": now(), "selection_sha256": check.sha(out / "COMPOSED_SELECTION.json"),
              "evaluation_started_utc": evaluation_started, "complete_matrix": len(complete) == 12,
              "witness_count": len(rows), "selection_groups": len(choices), "raw_metrics": rows,
              "runtime_seconds": time.perf_counter() - started,
              "scope": "Every eligible K-C/K-D and three-probability bank candidate is retained; F excluded; fixed public parameter composition on a new example",
              "limitations": ["Expanded model family and representation-fitting label exposure, not a matched-budget teacher audit.",
                              "Epoch-zero catch-up winners can retain the incoming observer without additional optimization improving them.",
                              "Prediction composition does not create information absent from the teacher; this is not a training-data privacy statement."]}
    with (out / "COMPOSED_ATTACKS.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    fields = ["teacher", "seed", "source_release", "bank", "target", "audit_budget", "candidate_id", "scope", "selected_epoch", "selected_epoch_zero",
              "validation_log_loss", "test_log_loss", "test_person_weighted_log_loss", "global_independent", "global_catchup", "global_pooled", "witness_id"]
    with (out / "COMPOSED_ATTACKS.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fields if key in row}
            record.update(test_log_loss=row["test"]["log_loss"], test_person_weighted_log_loss=row["test_person_weighted"]["log_loss"],
                          **{"global_" + scope: flag for scope, flag in row["selected_global"].items()})
            writer.writerow(record)
    lines = ["# Composed teacher attacks", "", "All retained witnesses are frozen teacher-only K-C/K-D models or three-probability source banks followed by their existing auditors. Full-input F models are ineligible. The separately frozen choice uses attacker-validation loss only; person weights score those same choices.", "",
             f"Retained {len(rows)} candidate witnesses in {len(choices)} teacher/seed/attribute/budget groups. The standalone saved-observer row is diagnostic; epoch zero remains eligible inside catch-up. Full candidate scores and exposure are in [COMPOSED_ATTACKS.csv](COMPOSED_ATTACKS.csv) and [COMPOSED_ATTACKS.json](COMPOSED_ATTACKS.json).", "",
             "Each mapper has 4,176 stored parameters, including 2,048 inactive raw-input weights; its effective teacher-only mapper has 2,128 parameters and one hidden ReLU layer. Banks add 51 source-head parameters and three sigmoid outputs. The unused 3,168-parameter decoder never enters a predictive composition. JSON records the actual auditor coefficient counts or tree/leaves/depth and all preprocessing stages per witness.", "",
             "For fixed public parameters on a new example, a(g(T)) equals the stored-release auditor. This extends the teacher witness family with mapper depth and source-label training; D and inherited observers add distinct real-protected-label exposure. It does not replace the teacher's standard direct audit, establish matched budgets, or create information absent from the teacher.", "",
             "| Teacher | Seed | Attribute | Budget | Scope | Witness | Development loss | PWGTP loss |", "|---|---:|---|---:|---|---|---:|---:|"]
    lookup = {row["witness_id"]: row for row in rows}
    for key, selected in choices.items():
        for scope, wid in selected.items():
            if wid is None:
                continue
            row = lookup[wid]
            lines.append(f"| {row['teacher']} | {row['seed']} | {row['target']} | {row['audit_budget']} | {scope} | {row['source_release']}/{row['candidate_id']} | {row['test']['log_loss']:.6f} | {row['test_person_weighted']['log_loss']:.6f} |")
    with (out / "COMPOSED_ATTACKS.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = compose(args.out.resolve())
    print(json.dumps({key: result[key] for key in ("witness_count", "selection_groups", "runtime_seconds", "complete_matrix")}))


if __name__ == "__main__":
    main()
