"""Independent restricted-input, bank, utility and composed-auditor replay.

No scientific fitter is called. Literal tensor inference, independent raw-label
metrics and actual saved optimizer/RNG evidence check the frozen computation.
Teacher maps and affine coefficients use the prior read-only replay mechanism.
Training gradients and exact forks have a separate verifier.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_preservation_extended import checkpoint_evidence
from scripts.verify_acs_selective import map_probability_free, verify_teachers, verify_geometry, teacher_components, geometry_score

SOURCES = ("income_binary", "civilian_at_work", "public_coverage")


def unit_path(out, unit):
    return out/f"{unit[0]}_{unit[1]}"/f"seed_{unit[-1]}"


def literal_release(state, teacher, *, access, raw=None, bank=False):
    """Independent 48-column forward; K cannot receive a raw feature array."""
    teacher = np.asarray(teacher, np.float32)
    if teacher.ndim != 2 or teacher.shape[1] != 16 or not np.isfinite(teacher).all():
        raise ValueError("Expected finite teacher16")
    mean, scale = (state[key].numpy() for key in ("input_mean", "input_scale"))
    u = ((teacher.astype(np.float64)-mean[:16])/scale[:16]).astype(np.float32)
    if access == "K":
        if raw is not None:
            raise ValueError("K replay rejects raw data")
        v = np.zeros((len(teacher), 32), np.float32)
    elif access == "F":
        if raw is None or np.shape(raw) != (len(teacher), 32):
            raise ValueError("F replay requires aligned raw PCA32")
        v = ((np.asarray(raw, np.float64)-mean)/scale).astype(np.float32)
    else:
        raise ValueError("Unknown access")
    x = torch.from_numpy(np.column_stack((u, v)))
    with torch.no_grad():
        h = F.linear(torch.relu(F.linear(x, state["mapper.0.weight"], state["mapper.0.bias"])),
                     state["mapper.2.weight"], state["mapper.2.bias"])
        native = {}
        for target in SOURCES:
            p = torch.sigmoid(F.linear(h, state[f"heads.{target}.weight"], state[f"heads.{target}.bias"])).reshape(-1)
            native[target] = torch.stack((1.-p, p), 1).numpy()
        release = np.column_stack([native[target][:, 1] for target in SOURCES]).astype(np.float32) if bank else h.numpy().astype(np.float32)
    return release, native


def audit_path(directory, key, cid, budget):
    path = directory/"fitted"/key
    if cid == "saved_adversary":
        return path/"saved_start/saved"
    return path/("saved_start" if cid == "catchup" else "fresh")/f"nested{budget}"/cid


def verify_releases(directory, unit, releases, pca, maps, report):
    frozen = check.read(directory/"release_freeze.json")
    state_paths = {"I": "initialization.pt", "W": "warm_base.pt", "C": "C/final.pt", "D": "D/final.pt", "B": "B/final.pt"}
    states = {}
    is_bank = unit[1] == "bank"
    for name, arrays in releases.items():
        state = torch.load(directory/"training"/state_paths[name], map_location="cpu", weights_only=True)["model_state"]
        assert check.state_hash(state) == frozen["state"]["models"][name]
        assert state["mapper.0.weight"].shape == (64, 48)
        if unit[1] != "F":
            assert torch.count_nonzero(state["mapper.0.weight"][:, 16:]) == 0
        states[name] = state
        for pool, raw in pca.items():
            teacher = map_probability_free(maps[unit[0]], raw[:, :16])
            kwargs = {"raw": raw} if unit[1] == "F" else {}
            release, _ = literal_release(state, teacher, access="F" if unit[1] == "F" else "K", bank=is_bank, **kwargs)
            np.testing.assert_array_equal(release, arrays[pool])
            assert release.shape[1] == (3 if is_bank else 16)
            report["literal_model_release_arrays"] += 1
    initial = torch.load(directory/"training/initialization.pt", map_location="cpu", weights_only=True)["model_state"]
    for pool, raw in pca.items():
        teacher = map_probability_free(maps[unit[0]], raw[:, :16])
        kwargs = {"raw": raw} if unit[1] == "F" else {}
        release, _ = literal_release(initial, teacher, access="F" if unit[1] == "F" else "K", **kwargs)
        np.testing.assert_allclose(release, teacher, atol=1e-5, rtol=1e-5)
        error = release.astype(float)-teacher.astype(float)
        if pool == "test":
            parity = check.read(directory/"initial_evaluation_parity.json")
        else:
            parity = frozen["initialization_identity"]["teacher_output_parity"][pool]
            assert parity["teacher_sha256"] == check.array_hash(teacher)
            assert parity["initial_sha256"] == check.array_hash(release)
        check.compare(float(np.max(np.abs(error))), parity["max_abs"], f"{unit}/{pool}/initial_parity_max")
        check.compare(float(np.sqrt(np.mean(error**2))), parity["rms"], f"{unit}/{pool}/initial_parity_rms")
    return states


def verify_native(directory, unit, states, pca, maps, frames, report):
    recorded = check.read(directory/"native_source.json")
    assert recorded["output_order"] == list(SOURCES)
    assert len(recorded["raw_metrics"]) == len(states)*2*3
    with np.load(directory/"native_predictions.npz") as saved:
        assert len(saved.files) == len(states)*2*3
        for name, state in states.items():
            for pool, split in (("source_validation", "source_validation"), ("test", "test")):
                teacher = map_probability_free(maps[unit[0]], pca[pool][:, :16])
                kwargs = {"raw": pca[pool]} if unit[1] == "F" else {}
                _, native = literal_release(state, teacher, access="F" if unit[1] == "F" else "K", **kwargs)
                for target in SOURCES:
                    expected = native[target]
                    np.testing.assert_array_equal(expected, saved[f"{split}/{name}/{target}"])
                    row, = [r for r in recorded["raw_metrics"] if (r["release"], r["target"], r["split"]) == (name, target, split)]
                    assert row["probability_sha256"] == check.array_hash(expected)
                    y, mask = check.labels(frames[pool], target)
                    for field, weight in (("score", None), ("person_weighted", frames[pool].PWGTP.to_numpy(float)[mask])):
                        check.compare(check.independent_scores(y[mask], expected[mask], 2, weight), row[field], f"{unit}/native/{split}/{name}/{target}/{field}")
                        report["native_score_dictionaries"] += 1
                    report["native_prediction_arrays"] += 1


def verify_composition_parity(directory, unit, states, releases, pca, maps, frames, selection, report):
    records = check.read(directory/"composition_parity.json")
    assert records["eligible"] == (unit[1] != "F")
    if unit[1] == "F":
        assert records["all_existing_audit_candidates"] == []
        return
    expected = {(budget, key.split('/')[1], key.split('/')[2], cid): metadata
                for budget in (120, 360)
                for key, fitted in selection['budgets'][str(budget)]['fitting_records'].items()
                for cid, metadata in fitted['candidates'].items()}
    recorded = {(row['audit_budget'], row['source_release'], row['target'], row['candidate_id']): row
                for row in records['all_existing_audit_candidates']}
    assert len(recorded) == len(records['all_existing_audit_candidates'])
    assert set(recorded) == set(expected)
    for key, metadata in expected.items():
        budget, name, target, cid = key
        record = recorded[key]
        assert record['seed'] == unit[-1] and record['teacher'] == unit[0]
        assert record['bank'] == (unit[1] == 'bank')
        assert not record['raw_input_argument_supplied'] and record['teacher_only_composition']
        assert record['model_state_sha256'] == check.state_hash(states[name])
        predict, _, _, _ = check.load_inference(audit_path(directory, f'audit/{name}/{target}', cid, budget), metadata)
        for split, pool in (('validation', 'attacker_validation'), ('test', 'test')):
            teacher = map_probability_free(maps[unit[0]], pca[pool][:, :16])
            composed_release, _ = literal_release(states[name], teacher, access='K', bank=unit[1] == 'bank')
            # No raw array enters either teacher-only inference call. Perturbing
            # an external raw-row array cannot affect this serialized function.
            external_raw = pca[pool][::-1].copy()
            external_raw[:] = 7.
            again, _ = literal_release(states[name], teacher, access='K', bank=unit[1] == 'bank')
            np.testing.assert_array_equal(composed_release, again)
            _, valid = check.labels(frames[pool], target)
            direct, composed = predict(releases[name][pool][valid]), predict(composed_release[valid])
            parity = composition_comparison(composed, direct)
            check.compare(parity, record['splits'][split], f'{unit}/{key}/{split}/composition')
            with np.load(directory/'predictions.npz') as saved:
                np.testing.assert_array_equal(direct, saved[f'budget{budget}/{split}/audit/{name}/{target}/{cid}'])
        report['composition_witnesses'] += 1


def composition_comparison(composed, direct):
    """Full arrays, including unsupported classes; no reduced-score proxy."""
    np.testing.assert_allclose(composed, direct, atol=1e-5, rtol=1e-5)
    return {'bitwise_equal': bool(np.array_equal(composed, direct)),
            'max_abs': float(np.max(np.abs(composed-direct))),
            'rms': float(np.sqrt(np.mean((composed-direct)**2))),
            'composed_probability_sha256': check.array_hash(composed),
            'direct_probability_sha256': check.array_hash(direct)}


def verify_accounting(accounting, checkpoint, candidate_path, metadata, bank, cid):
    """Inspect frozen tensor/sklearn structure independently of its exporter."""
    import joblib
    state = checkpoint['model_state']
    sizes = {prefix: sum(v.numel() for key, v in state.items() if key.startswith(prefix+'.'))
             for prefix in ('mapper', 'heads', 'decoder')}
    assert sizes == {'mapper': 4176, 'heads': 51, 'decoder': 3168}
    assert accounting['raw_input_coordinates_supplied'] == 0
    assert accounting['mapper']['inactive_raw_weights'] == 2048
    assert accounting['mapper']['stored_parameters'] == sizes['mapper']
    assert accounting['mapper']['effective_parameters'] == sizes['mapper']-2048
    assert accounting['mapper']['affine_layers'] == 2 and accounting['mapper']['hidden_relu_layers'] == 1
    assert accounting['source_heads']['stored_parameters'] == 51
    assert accounting['source_heads']['used_in_predictive_composition'] == bank
    assert accounting['decoder']['stored_parameters'] == 3168
    assert not accounting['decoder']['used_in_training_or_inference']
    assert accounting['frozen_model_inference_parameters_stored'] == 4176+(51 if bank else 0)
    assert accounting['frozen_model_inference_parameters_effective'] == 2128+(51 if bank else 0)
    assert accounting['auditor_fixed_parameters'] == metadata['parameters']
    auditor = accounting['auditor']
    if metadata['family'] == 'mlp':
        saved = torch.load(candidate_path/'model.pt', weights_only=True, map_location='cpu')['state']
        assert auditor['trainable_numeric_parameters'] == sum(v.numel() for v in saved.values())
        assert auditor['affine_layers'] == 3 and auditor['hidden_relu_layers'] == 2
        assert auditor['layer_widths'] == [metadata['input_dim'], 64, 32, metadata['n_classes']]
    else:
        model = joblib.load(candidate_path/'model.joblib')
        if metadata['family'] == 'logistic':
            assert auditor['trainable_numeric_parameters'] == model.coef_.size+model.intercept_.size
            assert auditor['coefficient_shape'] == list(model.coef_.shape)
        else:
            trees = [tree for iteration in model._predictors for tree in iteration]
            assert auditor['trainable_numeric_parameters'] is None
            assert auditor['trees'] == len(trees)
            assert auditor['nodes'] == sum(len(tree.nodes) for tree in trees)
            assert auditor['leaves'] == sum(int(tree.nodes['is_leaf'].sum()) for tree in trees)
            assert auditor['maximum_tree_depth'] == max(int(tree.nodes['depth'].max()) for tree in trees)
    pre = accounting['preprocessing']
    assert pre['original_mean_sha256'] == check.array_hash(state['input_mean'].numpy())
    assert pre['original_scale_sha256'] == check.array_hash(state['input_scale'].numpy())
    with np.load(candidate_path/'preprocessing.npz') as arrays:
        for key in ('mean', 'scale'):
            assert pre['auditor_'+key+'_sha256'] == check.array_hash(arrays[key])
            np.testing.assert_array_equal(pre['auditor_'+key], arrays[key])
    if cid in ('saved_adversary', 'catchup'):
        np.testing.assert_array_equal(pre['auditor_mean'], np.zeros(16))
        np.testing.assert_array_equal(pre['auditor_scale'], np.ones(16))


def verify_global_composition(out, cfg, units):
    if not (out/'COMPOSED_SELECTION.json').exists():
        assert len(units) < len(cfg['execution_order']), 'Complete matrix requires composed witness selection before final replay'
        return {'run': False, 'reason': 'First completed-block plumbing replay; global choices await the complete scientific matrix'}
    selected = check.read(out/'COMPOSED_SELECTION.json')
    result = check.read(out/'COMPOSED_ATTACKS.json')
    assert selected['protocol_freeze_sha256'] == check.sha(out/'protocol_freeze.json')
    assert selected['script_sha256'] == check.sha(ROOT/'scripts/compose_acs_restricted_attacks.py')
    assert not selected['development_predictions_opened']
    assert not selected['standalone_saved_adversary_eligible'] and selected['catchup_epoch_zero_eligible']
    assert selected['created_utc'] <= result['evaluation_started_utc']
    assert result['selection_sha256'] == check.sha(out/'COMPOSED_SELECTION.json')
    eligible_units = [unit for unit in units if unit[1] != 'F']
    assert selected['completed_eligible_units'] == eligible_units
    assert selected['complete_matrix'] == result['complete_matrix'] == (len(eligible_units) == 12)
    lookup = {row['witness_id']: row for row in selected['witnesses']}
    assert len(lookup) == len(selected['witnesses']) == result['witness_count']
    assert len(result['raw_metrics']) == len(lookup)
    expected_ids, independent_choices = set(), {}
    for unit in eligible_units:
        directory = unit_path(out, unit)
        relative = str(directory.relative_to(out))
        for name, expected in selected['unit_manifests_sha256'][relative].items():
            assert check.sha(directory/name) == expected
        selection = check.read(directory/'selection_before_test.json')
        training = check.read(directory/'training/training.json')
        support = check.read(directory/'support.json')
        metrics = check.read(directory/'metrics.json')
        metric_lookup = {(r['audit_budget'], r['release'], r['target'], r['candidate_id']): r
                         for r in metrics['raw_metrics'] if r['role'] == 'audit'}
        for budget in (120, 360):
            records = selection['budgets'][str(budget)]
            for key, fitted in records['fitting_records'].items():
                _, release, target = key.split('/')
                group = f'{unit[0]}/seed_{unit[-1]}/{target}/budget{budget}'
                groups = independent_choices.setdefault(group, {'independent': [], 'catchup': [], 'pooled': []})
                for cid, metadata in fitted['candidates'].items():
                    wid = f'{relative}/{key}/budget{budget}/{cid}'
                    expected_ids.add(wid)
                    witness = lookup[wid]
                    assert witness['unit'] == unit and witness['directory'] == relative
                    assert witness['validation_log_loss'] == metadata['validation_scores']['log_loss']
                    assert witness['validation_scores'] == metadata['validation_scores']
                    assert witness['validation_feature_label_sha256'] == metadata['validation_hashes']
                    assert witness['attacker_validation_pool_raw_rows_sha256'] == support['attacker_validation']['raw_row_sha256']
                    assert witness['attacker_fitting_subset_raw_rows_sha256'] == selection['audit_fit_indices'][target]['raw_rows_sha256']
                    assert witness['eligible'] == (cid != 'saved_adversary')
                    assert witness['selected_epoch_zero'] == (metadata.get('selected_epoch') == 0)
                    assert witness['model_checkpoint_sha256'] == check.sha(ROOT/witness['model_checkpoint'])
                    checkpoint = torch.load(ROOT/witness['model_checkpoint'], weights_only=True, map_location='cpu')
                    assert witness['model_state_sha256'] == check.state_hash(checkpoint['model_state'])
                    verify_accounting(witness['parameter_depth_preprocessing'], checkpoint,
                                      ROOT/witness['auditor_directory'], metadata, unit[1] == 'bank', cid)
                    for filename, digest in witness['auditor_files_sha256'].items():
                        assert digest == check.sha(ROOT/witness['auditor_directory']/filename)
                    exposure = witness['exposure']
                    assert exposure['source_model'] == training['native_source_head_training_exposure']
                    assert exposure['source_model_label_sha256'] == training['source_label_hashes']
                    assert exposure['real_protected_mapper_gradient'] == (release == 'D')
                    assert exposure['real_protected_mapper_epochs'] == (80 if release == 'D' else 0)
                    assert exposure['observer_parameters_in_predictive_composition'] == (cid in ('saved_adversary', 'catchup'))
                    assert exposure['inherited_catchup'] == metadata.get('inherited_exposure')
                    expected_steps = training['arms'][release]['optimizer_counts_including_common']['adversary_optimizer_steps']
                    assert exposure['observer_optimizer_steps'] == expected_steps
                    if unit[1] == 'bank':
                        assert expected_steps == exposure['observer_row_exposures'] == 0
                    item = (metadata['validation_scores']['log_loss'], wid)
                    if cid != 'saved_adversary':
                        groups['pooled'].append(item)
                        groups['catchup' if cid == 'catchup' else 'independent'].append(item)
                    output, = [row for row in result['raw_metrics'] if row['witness_id'] == wid]
                    source = metric_lookup[budget, release, target, cid]
                    for field in ('test', 'test_person_weighted', 'validation_person_weighted'):
                        assert output[field] == source[field]
    assert set(lookup) == expected_ids
    choices = {key: {scope: min(values)[1] if values else None for scope, values in groups.items()}
               for key, groups in independent_choices.items()}
    assert choices == selected['choices']
    for row in result['raw_metrics']:
        group = f"{row['teacher']}/seed_{row['seed']}/{row['target']}/budget{row['audit_budget']}"
        assert row['selected_global'] == {scope: wid == row['witness_id'] for scope, wid in choices[group].items()}
    return {'passed': True, 'witnesses': len(lookup), 'choice_groups': len(choices),
            'all_F_excluded': True, 'all_K_main_and_bank_candidates_retained': True,
            'global_choices_recomputed_only_from_validation': True,
            'separate_source_observer_attacker_exposure_checked': True,
            'selection_sha256': check.sha(out/'COMPOSED_SELECTION.json')}


def verify_bank_geometry(study, pcas, maps, preprocessing):
    """Independent SVD replay of the fixed six published-bank supplement."""
    out = study/'bank_geometry'
    report = check.read(out/'geometry.json')
    protocol, frozen, cfg = (check.read(out/name) for name in ('protocol_freeze.json', 'affine_freeze.json', 'config.json'))
    assert protocol['original_protocol_freeze_sha256'] == check.sha(study/'protocol_freeze.json')
    assert not protocol['any_fits_started'] and not frozen['heldout_evaluation_started']
    assert protocol['created_utc'] <= frozen['created_utc'] <= report['evaluation_started_utc']
    for field in ('source_and_config_sha256', 'required_input_sha256'):
        for filename, expected in protocol[field].items():
            assert check.sha(ROOT/filename) == expected
    assert cfg['number_of_fits'] == frozen['number_of_fits'] == report['number_of_fits'] == 18
    assert cfg['targets'] == ['raw', 'E', 'qE']
    assert cfg['units'] == [[teacher, seed] for teacher in ('E', 'S') for seed in (0, 1, 2)]
    assert report['input_dimension'] == 3 and report['target_dimension'] == 16
    assert report['rcond'] == report['variance_floor'] == 1e-12
    assert not report['hidden_bank_mapper_release_received'] and not report['outcome_labels_received']
    assert report['affine_freeze_sha256'] == check.sha(out/'affine_freeze.json')
    assert report['protocol_freeze_sha256'] == frozen['protocol_freeze_sha256'] == check.sha(out/'protocol_freeze.json')
    for filename, expected in check.read(out/'completion.json')['sha256'].items():
        assert check.sha(out/filename) == expected
    count, max_coefficient_error, max_prediction_error = 0, 0., 0.
    for key, snapshot in report['snapshots'].items():
        teacher, access, seed = snapshot['unit']
        assert key == f'{teacher}_bank/seed_{seed}' and access == 'bank'
        assert snapshot['direct_coordinate_error'] is None and snapshot['input_dimension'] == 3
        with np.load(study/key/'release_B.npz') as arrays:
            bank = {pool: arrays[pool].copy() for pool in arrays.files}
        mean, scale = (np.asarray(preprocessing[seed][field])[:16] for field in ('mean', 'scale'))
        targets = {pool: teacher_components(raw[:, :16], maps[seed], mean, scale) for pool, raw in pcas[seed].items()}
        x = bank['representation_fit'].astype(np.float64)
        assert x.shape[1] == 3 and np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
        design = np.column_stack((x, np.ones(len(x))))
        left, singular, right = np.linalg.svd(design, full_matrices=False)
        retained = singular > singular[0]*1e-12
        for target, record in snapshot['targets'].items():
            assert target in ('raw', 'E', 'qE')
            frozen_record = frozen['snapshots'][key]['targets'][target]
            assert all(record[field] == value for field, value in frozen_record.items())
            artifact = out/record['artifact']
            assert check.sha(artifact) == record['artifact_sha256'] == frozen['files_sha256'][record['artifact']]
            with np.load(artifact) as arrays:
                coefficient, prior, variance = (arrays[field].copy() for field in ('coefficient', 'prior', 'fit_variance'))
            y = targets['representation_fit'][target]
            assert coefficient.shape == (4, 16)
            assert check.array_hash(coefficient) == record['coefficient_sha256']
            assert check.array_hash(x) == record['fit_release_sha256']
            assert check.array_hash(y) == record['fit_target_sha256']
            np.testing.assert_array_equal(prior, y.mean(0))
            np.testing.assert_array_equal(variance, np.var(y, axis=0))
            np.testing.assert_array_equal(prior, record['prior'])
            np.testing.assert_array_equal(variance, record['fit_variance'])
            alternative = (right[retained].T/singular[retained]) @ (left[:, retained].T @ y)
            np.testing.assert_allclose(coefficient, alternative, atol=1e-8, rtol=1e-10)
            error = float(np.max(np.abs(coefficient-alternative)))
            prediction_error = float(np.max(np.abs(design@coefficient-design@alternative)))
            assert prediction_error < 1e-8
            max_coefficient_error, max_prediction_error = max(max_coefficient_error, error), max(max_prediction_error, prediction_error)
            assert record['rank'] == record['design_rank'] == int(retained.sum())
            np.testing.assert_allclose(record['singular_values'], singular, atol=1e-10, rtol=1e-12)
            for field, values in (('release_centered_rank', x-x.mean(0)), ('target_centered_rank', y-prior), ('target_uncentered_rank', y)):
                spectrum = np.linalg.svd(values, compute_uv=False)
                assert record[field]['rank'] == int((spectrum > spectrum[0]*1e-12).sum())
                np.testing.assert_allclose(record[field]['singular_values'], spectrum, atol=1e-10, rtol=1e-12)
            for pool, field in (('representation_fit', 'fit'), ('source_validation', 'source_validation'), ('test', 'development_evaluation')):
                scores = geometry_score(coefficient, prior, variance, bank[pool], targets[pool][target])
                for name, expected in scores.items():
                    check.compare(expected, record[field][name], f'bank_geometry/{key}/{target}/{field}/{name}')
                values = targets[pool][target]
                spectrum = np.linalg.svd(values-values.mean(0), compute_uv=False)
                assert record[field]['evaluation_target_centered_rank']['rank'] == int((spectrum > spectrum[0]*1e-12).sum())
            count += 1
    assert count == 18
    return {'passed': True, 'affine_decoders': count, 'published_bank_dimension': 3,
            'target_dimension': 16, 'max_independent_SVD_coefficient_error': max_coefficient_error,
            'max_independent_SVD_fitting_prediction_error': max_prediction_error,
            'all_fit_source_validation_development_scores_replayed': True,
            'hidden_bank_release_read': False, 'direct_coordinate_matching_undefined': True,
            'geometry_sha256': check.sha(out/'geometry.json')}


def verify(out):
    started = time.perf_counter()
    check.errors.clear()
    check.max_error, check.max_error_path, check.numeric_comparisons = 0., None, 0
    cfg = check.read(out/"config.json")
    freeze = check.read(out/"protocol_freeze.json")
    for field in ("sha256", "reference_record_hashes", "required_local_reference_hashes"):
        for name, expected in freeze[field].items():
            assert check.sha(ROOT/name) == expected, (field, name)
    parent = ROOT/cfg["parent_results"]
    pcfg = check.read(parent/"config.json")
    raw_path = ROOT/pcfg["raw_path"]
    assert check.sha(raw_path) == check.read(parent/"schema_support.json")["raw_sha256"]
    raw = pd.read_csv(raw_path, usecols=["MIG", "JWMNP", "PINCP", "ESR", "PUBCOV", "SEX", "RAC1P", "PWGTP"], low_memory=False)
    units = [unit for unit in cfg["execution_order"] if (unit_path(out, unit)/"metrics.json").exists()]
    assert units
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Independent restricted input, native source, composed-auditor, saved-model, raw-label metric and affine replay; no fitting or new data",
        "evaluation_status": cfg["evaluation_status"], "script_sha256": check.sha(__file__),
        "independent_metric_helper_sha256": check.sha(check.__file__), "absolute_metric_tolerance": check.TOL,
        "complete_matrix": len(units) == len(cfg["execution_order"]), "completed_units": units,
        "candidate_records": 0, "score_dictionaries": 0, "prediction_sets": 0, "selection_records": 0,
        "audit_terminal_checkpoints": 0, "nested_prefix_checks": 0, "mlp_schedule_checks": 0,
        "saved_direct_coordinate_fidelity": 0, "teacher_seeds": {}, "units": {},
        "literal_model_release_arrays": 0, "native_score_dictionaries": 0, "native_prediction_arrays": 0, "composition_witnesses": 0,
        "limitations": ["Recorded optimizer actions are inspected through state/counters/schedules, not rerun.",
                       "Residual components are not pure sensitive information; high affine error does not preclude nonlinear recovery."]}
    maps_by_seed, pca_by_seed, frames_by_seed, pre_by_seed, rows_by_seed = {}, {}, {}, {}, {}
    for seed in cfg["seeds"]:
        if not any(unit[-1] == seed for unit in units):
            continue
        directory = ROOT/cfg["selective_reference_results"]/"static"/f"seed_{seed}"
        with np.load(directory/"split_rows.npz") as arrays:
            rows = {key: arrays[key].copy() for key in arrays.files}
        with np.load(parent/f"seed_{seed}"/"split_rows.npz") as original:
            assert set(rows) == set(original.files)
            assert all(np.array_equal(value, original[key]) for key, value in rows.items())
        frames = {pool: raw.iloc[ix] for pool, ix in rows.items()}
        with np.load(ROOT/cfg["reference_results"]/f"seed_{seed}"/"release_E_pca.npz") as arrays:
            pca = {key: arrays[key].copy() for key in arrays.files}
        labels = {}
        for target in ("SEX", "RAC1P"):
            y, valid = check.labels(frames["representation_fit"], target)
            labels[target] = np.where(valid, y, -1).astype(np.int64)
        maps_by_seed[seed], report["teacher_seeds"][str(seed)] = verify_teachers(directory/"teachers", pca, labels, rows["representation_fit"], seed)
        pre_by_seed[seed] = check.read(ROOT/cfg["init_reference_results"]/f"seed_{seed}"/"training/training.json")["preprocessing"]
        pca_by_seed[seed], frames_by_seed[seed], rows_by_seed[seed] = pca, frames, rows
    for unit in units:
        directory, seed = unit_path(out, unit), unit[-1]
        measured, selection = check.read(directory/"metrics.json"), check.read(directory/"selection_before_test.json")
        frozen = check.read(directory/"release_freeze.json")
        provenance = check.read(directory/'parent_provenance.json')
        for field in ('used_reference_files_sha256', 'used_original_files_sha256'):
            for filename, digest in provenance[field].items():
                assert check.sha(ROOT/filename) == digest
        teacher_manifest = ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers/teachers.json'
        assert provenance['teacher_manifest_sha256'] == check.sha(teacher_manifest)
        assert selection["release_freeze_sha256"] == check.sha(directory/"release_freeze.json")
        assert selection["affine_freeze_sha256"] == check.sha(directory/"affine_freeze.json")
        assert selection["protocol_freeze_sha256"] == check.sha(out/"protocol_freeze.json")
        assert selection["created_utc"] <= measured["integrity"]["evaluation_started_utc"]
        assert all(value for key, value in measured["integrity"].items() if key.endswith("_unchanged"))
        for artifact in check.read(directory/"local_artifacts.json"):
            assert check.sha(ROOT/artifact["path"]) == artifact["sha256"]
        completion = check.read(directory/"completion.json")
        assert all(check.sha(directory/name) == expected for name, expected in completion["sha256"].items())
        with np.load(directory/"split_rows.npz") as rows:
            assert all(np.array_equal(rows[pool], ix) for pool, ix in rows_by_seed[seed].items())
        releases = {}
        for name in frozen["output_hashes"]:
            with np.load(directory/f"release_{name}.npz") as arrays:
                releases[name] = {pool: arrays[pool].copy() for pool in arrays.files}
            for pool, values in releases[name].items():
                expected = measured["integrity"]["evaluation_output_hashes"][name] if pool == "test" else frozen["output_hashes"][name][pool]
                assert check.array_hash(values) == expected
        geometry = verify_geometry(directory, releases, pca_by_seed[seed], pre_by_seed[seed], maps_by_seed[seed]) if unit[1] != "bank" else {"affine_decoders": 0, "scope": "bank publishes only three source probabilities"}
        states = verify_releases(directory, unit, releases, pca_by_seed[seed], maps_by_seed[seed], report)
        verify_native(directory, unit, states, pca_by_seed[seed], maps_by_seed[seed], frames_by_seed[seed], report)
        frames, checkpoints = frames_by_seed[seed], {}
        training_adversaries = {} if unit[1] == "bank" else {
            name: torch.load(directory/"training"/name/"final.pt", map_location="cpu", weights_only=True)["adversary_state"]
            for name in ("C", "D")}
        expected_rows = 30 if unit[1] == "bank" else 86
        assert len(measured["raw_metrics"]) == expected_rows
        with np.load(directory/"predictions.npz") as probabilities:
            assert len(probabilities.files) == expected_rows*2
            for row in measured["raw_metrics"]:
                role, release, target, cid = (row[field] for field in ("role", "release", "target", "candidate_id"))
                budget = row.get("audit_budget")
                record = selection if budget is None else selection["budgets"][str(budget)]
                prefix, key = ("" if budget is None else f"budget{budget}/"), f"{role}/{release}/{target}"
                metadata = record["fitting_records"][key]["candidates"][cid]
                classes = 9 if target == "RAC1P" else 2
                for split, pool in (("validation", "downstream_validation" if role == "transfer" else "attacker_validation"), ("test", "test")):
                    y, mask = check.labels(frames[pool], target)
                    probability = probabilities[f"{prefix}{split}/{key}/{cid}"]
                    for suffix, weights in (("", None), ("_person_weighted", frames[pool].PWGTP.to_numpy(float)[mask])):
                        check.compare(check.independent_scores(y[mask], probability, classes, weights), row[split+suffix],
                                      f"{unit}/{prefix}{key}/{cid}/{split}{suffix}")
                        report["score_dictionaries"] += 1
                check.compare(metadata["validation_scores"], row["validation"], f"{unit}/{key}/{cid}/fit_validation")
                assert row["selected"] == (record["head_selections"][key] == cid)
                assert row["independent_selected"] == (record["independent_selections"][key] == cid)
                assert row["selected_within_family"] == (record["family_selections"][key][row["family"]] == cid)
                assert row["auc_selected"] == (record["auroc_selections"][key] == cid)
                report["candidate_records"] += 1
            for budget, record in ((None, selection), (120, selection["budgets"]["120"]), (360, selection["budgets"]["360"])):
                for key, fitted in record["fitting_records"].items():
                    role, name, target = key.split("/")
                    classes = 9 if target == 'RAC1P' else 2
                    candidates = fitted["candidates"]
                    eligible = [cid for cid in candidates if cid != "saved_adversary"]
                    independent = [cid for cid in eligible if cid != "catchup"]
                    assert record["head_selections"][key] == check.primary(candidates, eligible)
                    assert record["independent_selections"][key] == check.primary(candidates, independent)
                    assert record["auroc_selections"][key] == check.primary(candidates, eligible, auc=True)
                    family = lambda cid: cid if cid in ("catchup", "saved_adversary") else candidates[cid]["family"]
                    for fam, cid in record["family_selections"][key].items():
                        assert cid == check.primary(candidates, [c for c in candidates if family(c) == fam])
                    report["selection_records"] += 1
                    fitpool, valpool = ("downstream_fit", "downstream_validation") if role == "transfer" else ("attacker_fit", "attacker_validation")
                    yf, mf = check.labels(frames[fitpool], target)
                    yv, mv = check.labels(frames[valpool], target)
                    j = (cfg["utility_tasks"] if role == "transfer" else ["SEX", "RAC1P"]).index(target)
                    fitseed, limit = ((1230000, 2048) if role == "transfer" else (1240000, 4096))
                    ix = np.random.default_rng(fitseed+100*seed+j).permutation(np.flatnonzero(mf))[:limit]
                    indexrecord = selection["task_fit_indices" if role == "transfer" else "audit_fit_indices"][target]
                    assert indexrecord["pool_indices_sha256"] == check.array_hash(ix)
                    assert indexrecord["raw_rows_sha256"] == check.array_hash(rows_by_seed[seed][fitpool][ix])
                    xf, xv = releases[name][fitpool][ix].astype(np.float64), releases[name][valpool][mv].astype(np.float64)
                    for cid, metadata in candidates.items():
                        expected_seed = (1250000 if role == 'transfer' else 1260000)+100*seed+j
                        if cid == 'mlp_1':
                            expected_seed += 10000
                        if cid == 'catchup':
                            expected_seed = 1300000+100*seed+j
                        if cid != 'saved_adversary':
                            assert metadata['seed'] == expected_seed
                        cpath = directory/"fitted"/key
                        if role == "transfer":
                            cpath /= cid
                        elif cid == "saved_adversary":
                            cpath = cpath/"saved_start"/"saved"
                        else:
                            cpath = cpath/("saved_start" if cid == "catchup" else "fresh")/f"nested{budget}"/cid
                        assert check.read(cpath/"metadata.json") == metadata
                        predict, mean, scale, state = check.load_inference(cpath, metadata)
                        assert metadata["validation_hashes"] == {"x": check.array_hash(xv), "y": check.array_hash(yv[mv].astype(np.int64))}
                        if cid != "saved_adversary":
                            assert metadata["fit_hashes"] == {"x": check.array_hash(xf), "y": check.array_hash(yf[ix].astype(np.int64))}
                        if cid in ("catchup", "saved_adversary"):
                            np.testing.assert_array_equal(mean, np.zeros(16))
                            np.testing.assert_array_equal(scale, np.ones(16))
                            saved = candidates["saved_adversary"]
                            final_training_state = {k.removeprefix(target+"."): value for k, value in training_adversaries[name].items() if k.startswith(target+".")}
                            assert saved["selected_state_hash"] == check.state_hash(final_training_state)
                            inherited = metadata["inherited_exposure"]
                            ty, tm = check.labels(frames["representation_fit"], target)
                            ty = np.where(tm, ty, -1).astype(np.int64)
                            assert inherited["training_label_sha256"] == check.array_hash(ty)
                            assert inherited["fit_raw_row_sha256"] == check.array_hash(rows_by_seed[seed]["representation_fit"])
                            assert inherited["total_row_exposures"] == len(ty)*260
                            assert inherited["total_optimizer_steps"] == int(np.ceil(len(ty)/256))*260
                            assert metadata["source_state_hash"] == saved["selected_state_hash"]
                            if cid == "saved_adversary":
                                assert np.array_equal(predict(xv), check.network_probability(state, xv))
                                report["saved_direct_coordinate_fidelity"] += 1
                            else:
                                assert metadata["initial_state_hash"] == saved["selected_state_hash"]
                                assert metadata["optimizer"]["initial_state_entries"] == 0
                                assert metadata["optimizer"]["restored_state"] is False
                        else:
                            np.testing.assert_array_equal(mean, xf.mean(0))
                            std = xf.std(0)
                            np.testing.assert_array_equal(scale, np.where(std > 1e-12, std, 1.))
                        prefix = "" if budget is None else f"budget{budget}/"
                        for split, pool in (("validation", valpool), ("test", "test")):
                            _, valid = check.labels(frames[pool], target)
                            assert np.array_equal(predict(releases[name][pool][valid]), probabilities[f"{prefix}{split}/{key}/{cid}"])
                            report["prediction_sets"] += 1
                        if "validation_curve" not in metadata:
                            continue
                        epochs = 40 if budget is None else budget
                        curve = metadata["validation_curve"]
                        assert metadata["parameters"]["epochs"] == epochs
                        assert [v["epoch"] for v in curve] == list(range(0, epochs+1, 5))
                        steps = int(np.ceil(len(ix)/256))
                        assert all(v["optimizer_steps"] == steps*v["epoch"] for v in curve)
                        best = min(curve, key=lambda v: (v["validation_log_loss"], v["epoch"]))
                        assert metadata["selected_epoch"] == best["epoch"]
                        assert metadata["validation_scores"]["log_loss"] == best["validation_log_loss"]
                        assert metadata["optimizer_steps"] == steps*epochs
                        assert metadata["training_row_exposures"] == len(ix)*epochs
                        check.replay_schedule(metadata, len(ix))
                        assert metadata['schedule_seed'] == expected_seed+700000
                        if cid != 'catchup':
                            with torch.random.fork_rng(devices=[]):
                                torch.manual_seed(expected_seed)
                                initial = torch.nn.Sequential(torch.nn.Linear(xf.shape[1], 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, classes))
                            assert check.state_hash(initial.state_dict()) == metadata['initial_state_hash']
                        report["mlp_schedule_checks"] += 1
                        if budget is None:
                            continue
                        terminal = directory/"fitted"/key/("saved_start" if cid == "catchup" else "fresh")/f"last_training_{cid}_epoch{budget}.pt"
                        checkpoints[f"{budget}/{key}/{cid}"] = checkpoint_evidence(terminal, metadata, len(ix))
                        report["audit_terminal_checkpoints"] += 1
                        if budget == 360:
                            lower = selection["budgets"]["120"]["fitting_records"][key]["candidates"][cid]
                            assert curve[:len(lower["validation_curve"])] == lower["validation_curve"]
                            assert metadata["validation_scores"]["log_loss"] <= lower["validation_scores"]["log_loss"]
                            report["nested_prefix_checks"] += 1
        report["units"][str(unit)] = {"passed": True, "metrics_sha256": check.sha(directory/"metrics.json"),
            "selection_sha256": check.sha(directory/"selection_before_test.json"), "predictions_sha256": check.sha(directory/"predictions.npz"),
            "geometry": geometry, "actual_training_checkpoints": checkpoints}
        verify_composition_parity(directory, unit, states, releases, pca_by_seed[seed], maps_by_seed[seed], frames_by_seed[seed], selection, report)
    report["global_composition_selection"] = verify_global_composition(out, cfg, units)
    if len(units) == len(cfg['execution_order']):
        report['bank_geometry_supplement'] = verify_bank_geometry(out, pca_by_seed, maps_by_seed, pre_by_seed)
    report.update(passed=not check.errors, errors=check.errors, numeric_values_compared=check.numeric_comparisons,
                  max_absolute_metric_error=check.max_error, max_error_path=check.max_error_path,
                  runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/redesign_20260908_acs_restricted_inputs_v1")
    parser.add_argument("--report", type=Path, help="Fresh report path; no overwrites; default stdout")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("Preserve existing verification evidence; use a fresh destination")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.report:
        with args.report.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
