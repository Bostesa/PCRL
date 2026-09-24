"""Branch A: calibrated T32 channel and bounded decoder/attack alternation.

Only the coordinator/execution owner dispatches this CLI.  All fitted objects
and person-derived coefficient arrays stay in a private unit directory.  The
outer development assessment is never loaded by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit
from experiments.pcrl_task_aligned_cuts_v1 import data, method, release, solver
from experiments.pcrl_task_directed_release_v1.audits import load_candidate
from . import alternate, reference, roles

PROTECTED_ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")
CLASS_COUNT = {"SEX": 2, "RAC1P": 9}
SCIENCE_ROLES = ("nuisance_train", "audit_fit", "coefficient_split",
                 "inner_selection", "inner_check")
MAX_ROUNDS = 6
PROBABILITY_FLOOR = method.PROBABILITY_FLOOR


def _sha_file(path):
    return data.sha256_file(path)


def _sha_array(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_or_verify_json(path, value):
    path = Path(path)
    encoded = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable unit record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(encoded)
    os.replace(temporary, path)


def _save_or_verify_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or any(
                not np.array_equal(prior[key], value) for key, value in arrays.items()):
                raise ValueError(f"immutable array archive differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _valid_mask(rows, target):
    y = np.asarray(rows["labels"][target])
    count = CLASS_COUNT[target] if target in CLASS_COUNT else 2
    if y.ndim != 1 or len(y) != len(rows["token_codes"]):
        raise ValueError("label/code alignment changed")
    return np.isfinite(y) & (y == np.floor(y)) & (y >= 0) & (y < count)


def aggregate_loss_coefficients(codes, per_person_token_loss, weights, n_states):
    """Group exact original-person token losses, normalized separately U/W."""
    t = np.asarray(codes)
    losses = np.asarray(per_person_token_loss, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (not isinstance(n_states, int) or n_states < 1 or t.ndim != 1
            or t.dtype.kind not in "iu" or len(t) == 0 or np.any(t < 0)
            or np.any(t >= n_states) or losses.ndim != 2 or losses.shape[0] != len(t)
            or losses.shape[1] < 1 or not np.isfinite(losses).all()
            or np.any(losses < 0) or w.shape != (len(t),)
            or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0):
        raise ValueError("aligned finite original-person code/loss/weight rows required")
    unweighted = np.zeros((n_states, losses.shape[1]), dtype=np.float64)
    weighted = np.zeros_like(unweighted)
    np.add.at(unweighted, t, losses / len(t))
    np.add.at(weighted, t, losses * (w / w.sum())[:, None])
    return {"U": unweighted, "W": weighted}


def source_laws(codes, q_ref, historical_q):
    """Initial H/D17/Q/coverage token laws on original people."""
    q_ref = method.validate_channel(q_ref, n_tokens=17)
    historical_q = method.validate_channel(historical_q, n_states=len(q_ref), n_tokens=17)
    t = np.asarray(codes)
    if t.ndim != 1 or t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t >= len(q_ref)):
        raise ValueError("stored T0 codes must index frozen channel states")
    return {"H": np.ones((len(t), 1), dtype=np.float64),
            "D17": q_ref[t], "Q": historical_q[t],
            "coverage": (q_ref[t] + historical_q[t] + 1/17) / 3}


def task_person_losses(decoder, rows, *, n_tokens=17):
    """Observed-class CE for every token, on eligible original people only."""
    valid = _valid_mask(rows, "same_residence")
    if not valid.any():
        raise ValueError("no observed task labels on declared role")
    h = np.asarray(rows["ha"], dtype=np.float64)
    if h.shape != (len(valid), 4) or not np.isfinite(h).all():
        raise ValueError("task decoder sees exactly four H_A coordinates")
    predictions = np.asarray(decoder.predict_token_proba(h, n_tokens), dtype=np.float64)
    if predictions.shape != (len(valid), n_tokens, 2):
        raise ValueError("frozen task decoder changed token/class schema")
    losses = method.observed_token_losses(
        np.asarray(rows["labels"]["same_residence"])[valid], predictions[valid],
        probability_floor=PROBABILITY_FLOOR)
    return valid, losses


def task_cost_pair(decoder, rows, n_states=32):
    """Frozen decoder U/W objective, with no second state-mass multiplication."""
    valid, losses = task_person_losses(decoder, rows)
    return aggregate_loss_coefficients(np.asarray(rows["token_codes"])[valid], losses,
                                       np.asarray(rows["weights"])[valid], n_states)


def task_scores(decoder, rows, q):
    """Exact-token task loss for a frozen decoder on one role."""
    q = method.validate_channel(q, n_tokens=17)
    valid, losses = task_person_losses(decoder, rows)
    t = np.asarray(rows["token_codes"])[valid]
    person = np.einsum("nz,nz->n", q[t], losses)
    w = np.asarray(rows["weights"])[valid]
    return {"U": float(person.mean()), "W": float(np.dot(w / w.sum(), person)),
            "n_original_people": int(len(person))}


def _population_hash(rows, target, valid):
    """Pin the *same* eligible original people for U and PWGTP cut matrices."""
    digest = hashlib.sha256()
    for name, value in (("ids", np.asarray(rows["ids"])[valid]),
                        ("codes", np.asarray(rows["token_codes"])[valid]),
                        ("labels", np.asarray(rows["labels"][target])[valid]),
                        ("weights", np.asarray(rows["weights"])[valid])):
        digest.update(name.encode())
        if name == "ids":
            for person_id in value:
                encoded = str(person_id).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        else:
            digest.update(bytes.fromhex(_sha_array(value)))
    return digest.hexdigest()


def _decoder_input_hash(rows, valid):
    digest = hashlib.sha256()
    for name, value in (("ha", rows["ha"]),
                        ("token_codes", rows["token_codes"]),
                        ("task_labels", rows["labels"]["same_residence"]),
                        ("weights", rows["weights"]),
                        ("households", rows["households"])):
        digest.update(name.encode())
        selected = np.asarray(value)[valid]
        if selected.dtype.kind in "OUS":
            for entry in selected:
                raw = str(entry).encode()
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        else:
            digest.update(bytes.fromhex(_sha_array(selected)))
    return digest.hexdigest()


def fit_frozen_decoder(role_dict, current_q, q_ref, output_dir, seed):
    """Fit/recover one task decoder slate on nuisance_train, select inner-only.

    Returns a hash-verified selected model for both Branch A and nested B. A
    complete slate without its small receipt can be recovered after a crash;
    a partial slate is preserved for diagnosis, never overwritten.
    """
    root = Path(output_dir)
    train = role_dict["nuisance_train"]
    selected = role_dict["inner_selection"]
    mask_train = _valid_mask(train, "same_residence")
    mask_select = _valid_mask(selected, "same_residence")
    if not mask_train.any() or not mask_select.any():
        raise ValueError("task decoder needs supported train and selection labels")
    inherited_audit.assert_household_disjoint(
        np.asarray(train["households"])[mask_train],
        np.asarray(selected["households"])[mask_select])
    current_q = method.validate_channel(current_q, n_tokens=17)
    q_ref = method.validate_channel(q_ref, n_states=len(current_q), n_tokens=17)
    expected = {"schema": 1, "seed": int(seed),
                "current_q_sha256": _sha_array(current_q),
                "q_ref_sha256": _sha_array(q_ref),
                "train_input_sha256": _decoder_input_hash(train, mask_train),
                "selection_input_sha256": _decoder_input_hash(selected, mask_select),
                "training_law": "(current_Q + D17 + uniform_17)/3",
                "fit_original_people": int(mask_train.sum()),
                "selection_original_people": int(mask_select.sum())}
    receipt_path = root / "FIT_DECODER.json"
    slate = root / "slate"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("existing decoder receipt differs from input contract")
    else:
        if not (slate / "slate.json").exists():
            if slate.exists() and any(slate.iterdir()):
                raise FileExistsError("partial task slate retained; quarantine before technical retry")
            alternate.fit_decoder_round(
                np.asarray(train["ha"])[mask_train],
                np.asarray(train["token_codes"])[mask_train],
                np.asarray(train["labels"]["same_residence"])[mask_train],
                np.asarray(train["weights"])[mask_train],
                np.asarray(selected["ha"])[mask_select],
                np.asarray(selected["token_codes"])[mask_select],
                np.asarray(selected["labels"]["same_residence"])[mask_select],
                np.asarray(selected["weights"])[mask_select],
                current_q, q_ref, seed=int(seed), out_dir=slate)
        slate_meta = json.loads((slate / "slate.json").read_text())
        cid = slate_meta["selection"]
        model_dir = slate / cid
        receipt = {**expected, "selected_candidate": cid,
                   "selected_model_relative_directory": str(model_dir.relative_to(root)),
                   "selected_model_sha256": inherited_audit.model_directory_hash(model_dir),
                   "slate_sha256": _sha_file(slate / "slate.json"),
                   "candidate_count": len(slate_meta["candidate_ids"])}
        _write_or_verify_json(receipt_path, receipt)
    if _sha_file(slate / "slate.json") != receipt["slate_sha256"]:
        raise ValueError("frozen task slate hash differs")
    model_dir = root / receipt["selected_model_relative_directory"]
    if inherited_audit.model_directory_hash(model_dir) != receipt["selected_model_sha256"]:
        raise ValueError("frozen task decoder model hash differs")
    return {**receipt, "model_directory": str(model_dir.resolve()),
            "decoder": load_candidate(model_dir), "receipt_sha256": _sha_file(receipt_path)}


def load_frozen_decoder(spec):
    """Load one pinned retained decoder without its training labels."""
    model_dir = Path(spec["model_directory"])
    if inherited_audit.model_directory_hash(model_dir) != spec["selected_model_sha256"]:
        raise ValueError("retained decoder hash mismatch")
    return load_candidate(model_dir)


def person_loss_from_attack(spec, coefficient_rows):
    """Replay a pinned fixed attack on original people for all 17 tokens.

    Branch B can regroup these person-level token losses under child IDs.  An
    H-only attack repeats its one prediction across all 17 token values.
    The arrays stay private and never become predictor inputs or wire fields.
    """
    target = spec["target"]
    if target not in CLASS_COUNT or spec["class_order"] != list(range(CLASS_COUNT[target])):
        raise ValueError("attack changed declared protected class order")
    model_dir = Path(spec["model_directory"])
    if inherited_audit.model_directory_hash(model_dir) != spec["model_sha256"]:
        raise ValueError("attack model object differs from frozen hash")
    candidate = load_candidate(model_dir)
    view, wire = spec["source_view"], spec["wire"]
    if wire not in ("H", "release") or (view == "B" and wire != "H"):
        raise ValueError("attack route receives illegal service/token view")
    h = inherited_audit.view_features(coefficient_rows["ha"],
                                      coefficient_rows["hb"], view)
    token_count = 1 if wire == "H" else 17
    if candidate.n_tokens != token_count or candidate.n_classes != CLASS_COUNT[target]:
        raise ValueError("frozen attack token/class schema mismatch")
    predictions = candidate.predict_token_proba(h, token_count)
    if predictions.shape != (len(h), token_count, CLASS_COUNT[target]):
        raise ValueError("frozen attack returned invalid prediction shape")
    if wire == "H":
        predictions = np.repeat(predictions, 17, axis=1)
    valid = _valid_mask(coefficient_rows, target)
    if not valid.any():
        raise ValueError("attack coefficient role lacks observed labels")
    losses = method.observed_token_losses(
        np.asarray(coefficient_rows["labels"][target])[valid], predictions[valid],
        probability_floor=PROBABILITY_FLOOR)
    if wire == "H" and not np.array_equal(losses, np.repeat(losses[:, :1], 17, axis=1)):
        raise AssertionError("H-only loss unexpectedly depends on token action")
    return valid, losses


def build_attack_bank(role_dict, source_channels, output_dir, seed,
                      *, target_roles=PROTECTED_ROLES, n_states=32):
    """Fit/recover fixed attacks and group same-row affine cut coefficients.

    Source channels are training laws, not additional released information.
    A and AB get legal H-only/token ancestors; B contributes only its H_B
    service to the coalition. Every frozen model is retained privately so a
    refined child partition can recompute exact person-token coefficients.
    """
    root = Path(output_dir)
    if not source_channels or not isinstance(source_channels, dict):
        raise ValueError("named attack-training source channels required")
    target_roles = tuple(target_roles)
    if not target_roles or len(set(target_roles)) != len(target_roles) or any(
            role not in PROTECTED_ROLES for role in target_roles):
        raise ValueError("unknown or duplicate protected target role")
    sources = {}
    for source, q in source_channels.items():
        if not isinstance(source, str) or not source:
            raise ValueError("source identity must be nonempty")
        if source == "H":
            if q is not None:
                raise ValueError("H-only source must not carry a token kernel")
            sources[source] = None
        else:
            sources[source] = method.validate_channel(q, n_states=n_states,
                                                      n_tokens=17)
    fit_rows, selection_rows = role_dict["audit_fit"], role_dict["inner_selection"]
    coefficient_rows = role_dict["coefficient_split"]
    inherited_audit.assert_household_disjoint(
        fit_rows["households"], selection_rows["households"],
        coefficient_rows["households"])
    fitted = {}
    targets = sorted({role.split("/")[1] for role in target_roles})
    counter = 0
    for target in targets:
        need_a = f"A/{target}" in target_roles or f"AB/{target}" in target_roles
        need_ab = f"AB/{target}" in target_roles
        for view in ("A", "AB", "B"):
            if view == "A" and not need_a or view in ("AB", "B") and not need_ab:
                continue
            for source, q in sources.items():
                if view == "B" and source != "H":
                    continue
                if q is None:
                    fit_law = np.ones((len(fit_rows["token_codes"]), 1))
                    select_law = np.ones((len(selection_rows["token_codes"]), 1))
                else:
                    fit_law = q[np.asarray(fit_rows["token_codes"], dtype=np.int64)]
                    select_law = q[np.asarray(selection_rows["token_codes"], dtype=np.int64)]
                role = f"attack:{view}/{target}"
                directory = root / "slates" / f"{view}_{target}" / source
                registry = inherited_audit.fit_role_slate(
                    fit_rows, fit_law, selection_rows, select_law, role,
                    directory, int(seed) + 1000 * counter,
                    release_id=source, slate="standard")
                fitted[(view, target, source)] = registry
                counter += 1
    specs, cuts = [], []
    for role in target_roles:
        role_view, target = role.split("/")
        allowed_views = ("A",) if role_view == "A" else ("A", "AB", "B")
        for source_view in allowed_views:
            for source in sources:
                key = (source_view, target, source)
                if key not in fitted:
                    continue
                registry = fitted[key]
                for cid, model in sorted(registry["models"].items()):
                    model_dir = Path(model["model_directory"]).resolve()
                    if not model_dir.is_relative_to(root.resolve()):
                        raise ValueError("fitted attack lies outside owned private unit")
                    wire = "H" if source == "H" else "release"
                    spec = {
                        "id": f"{role}/from_{source_view}/{source}/{cid}",
                        "role": role, "target": target, "source_view": source_view,
                        "wire": wire, "training_source": source,
                        "source_channel_sha256": "H_ONLY" if sources[source] is None else _sha_array(sources[source]),
                        "model_relative_directory": str(model_dir.relative_to(root.resolve())),
                        "model_directory": str(model_dir),
                        "model_sha256": model["model_sha256"],
                        "class_order": list(range(CLASS_COUNT[target])),
                        "fit_class_counts": registry["fit_class_counts"],
                        "fit_missing_classes": registry["fit_missing_classes"],
                    }
                    valid, losses = person_loss_from_attack(spec, coefficient_rows)
                    codes = np.asarray(coefficient_rows["token_codes"])[valid]
                    weight = np.asarray(coefficient_rows["weights"])[valid]
                    pair = aggregate_loss_coefficients(codes, losses, weight, n_states)
                    pool_hash = _population_hash(coefficient_rows, target, valid)
                    for weighting in ("U", "W"):
                        cuts.append({"id": spec["id"] + "/" + weighting,
                                     "role": role, "weighting": weighting,
                                     "coeff": pair[weighting],
                                     "coefficient_pool_sha256": pool_hash,
                                     "class_order": spec["class_order"],
                                     "weight_normalization": "1/n" if weighting == "U" else "PWGTP/sum(PWGTP)",
                                     "predictor_sha256": spec["model_sha256"],
                                     "source_view": source_view,
                                     "source_release": source,
                                     "model_candidate": cid})
                    specs.append(spec)
    if not cuts:
        raise ValueError("attack source fit produced no fixed cuts")
    archive = root / "coefficients.npz"
    _save_or_verify_npz(archive, **{f"cut_{i:04d}": cut["coeff"]
                                   for i, cut in enumerate(cuts)})
    manifest = {"schema": 1, "seed": int(seed), "target_roles": list(target_roles),
                "n_states": n_states, "n_tokens": 17,
                "source_channels_sha256": {name: "H_ONLY" if q is None else _sha_array(q)
                                           for name, q in sources.items()},
                "coefficient_archive_sha256": _sha_file(archive),
                "cuts": [{key: value for key, value in cut.items() if key != "coeff"}
                         for cut in cuts],
                "attack_specs": [{key: value for key, value in spec.items()
                                  if key != "model_directory"} for spec in specs],
                "fit_slate_count": len(fitted), "cut_count": len(cuts),
                "original_coefficient_people": int(len(coefficient_rows["token_codes"]))}
    _write_or_verify_json(root / "ATTACK_BANK.json", manifest)
    return {"cuts": cuts, "attack_specs": specs,
            "bank_manifest_sha256": _sha_file(root / "ATTACK_BANK.json"),
            "coefficient_archive_sha256": _sha_file(archive),
            "fit_slate_count": len(fitted)}


def load_frozen_bank(bank_dir):
    """Hash-check retained fitted attacks and rehydrate their private specs.

    Branch B uses the frozen models to recompute per-person losses under child
    state IDs; it must not regroup the already aggregated T32 matrices.
    """
    root = Path(bank_dir).resolve()
    manifest_path = root / "ATTACK_BANK.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or manifest.get("n_tokens") != 17:
        raise ValueError("unknown frozen attack-bank schema")
    archive = root / "coefficients.npz"
    if _sha_file(archive) != manifest["coefficient_archive_sha256"]:
        raise ValueError("frozen attack coefficient archive hash mismatch")
    cuts = []
    with np.load(archive, allow_pickle=False) as arrays:
        if set(arrays.files) != {f"cut_{i:04d}" for i in range(len(manifest["cuts"]))}:
            raise ValueError("frozen attack coefficient member set differs")
        for i, meta in enumerate(manifest["cuts"]):
            coeff = arrays[f"cut_{i:04d}"].copy()
            if coeff.shape != (manifest["n_states"], 17) or not np.isfinite(coeff).all():
                raise ValueError("frozen attack coefficient schema differs")
            cuts.append({**meta, "coeff": coeff})
    specs = []
    for meta in manifest["attack_specs"]:
        relative = Path(meta["model_relative_directory"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe frozen attack model path")
        model_dir = (root / relative).resolve()
        if not model_dir.is_relative_to(root) or inherited_audit.model_directory_hash(model_dir) != meta["model_sha256"]:
            raise ValueError("frozen attack model hash/path mismatch")
        specs.append({**meta, "model_directory": str(model_dir)})
    if len(cuts) != manifest["cut_count"] or len(specs)*2 != len(cuts):
        raise ValueError("frozen bank cut/model count differs")
    return {"cuts": cuts, "attack_specs": specs,
            "bank_manifest_sha256": _sha_file(manifest_path),
            "coefficient_archive_sha256": _sha_file(archive),
            "fit_slate_count": manifest["fit_slate_count"]}


def _role_fingerprint(rows):
    digest = hashlib.sha256()
    for name in ("ha", "hb", "token_codes", "weights", "ids", "households"):
        value = np.asarray(rows[name])
        digest.update(name.encode())
        if value.dtype.kind in "OUS":
            for entry in value:
                raw = str(entry).encode()
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        else:
            digest.update(bytes.fromhex(_sha_array(value)))
    for target in ("same_residence", "SEX", "RAC1P"):
        digest.update(target.encode())
        digest.update(bytes.fromhex(_sha_array(rows["labels"][target])))
    return digest.hexdigest()


def _inventory(root):
    return {str(path.relative_to(root)): _sha_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "COMPLETE.json"}


def _load_complete(root, expected):
    path = root / "COMPLETE.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("completed center unit differs from requested scientific unit")
    if receipt.get("artifact_sha256") != _inventory(root):
        raise ValueError("completed center unit artifact inventory differs")
    selected = receipt["selected_round"]
    artifact = release.ChannelArtifact.load(root / f"round_r{selected:02d}" / "channel")
    return {"status": "COMPLETE", "selected_channel": artifact.Q.copy(),
            "selected_round": selected, "rounds": receipt["rounds"],
            "outer_labels_accessed": False,
            "complete_receipt_sha256": _sha_file(path)}


def select_final_bank_feasible(channels, round_records, final_cost, final_cuts,
                               *, tolerance=solver.PRIMAL_TOL):
    """Choose only among channels satisfying the *last rebased union bank*.

    Earlier Q may have been feasible for its own bank but not for a later
    best-response cut. Its attractive inner-selection score cannot override
    that new constraint. Every old result remains in the final replay log.
    """
    if len(channels) != len(round_records) or not channels:
        raise ValueError("one retained channel required per round record")
    final_cost = np.asarray(final_cost, dtype=np.float64)
    checks = []
    for index, (q, record) in enumerate(zip(channels, round_records)):
        if record["round"] != index:
            raise ValueError("round records and channels misalign")
        replay = solver.replay_p1(q, final_cost, final_cuts)
        feasible = bool(replay["maximum_cut_violation"] <= tolerance and
                        replay["simplex_residual"] <= solver.SIMPLEX_TOL and
                        replay["minimum_entry"] >= -solver.NONNEGATIVE_TOL)
        checks.append({"round": index, "channel_sha256": _sha_array(q),
                       "final_bank_sha256": replay["bank_sha256"],
                       "maximum_cut_violation": replay["maximum_cut_violation"],
                       "simplex_residual": replay["simplex_residual"],
                       "feasible": feasible})
    eligible = [record for record, check in zip(round_records, checks) if check["feasible"]]
    if not eligible:
        raise RuntimeError("no fitted Q satisfies final rebased bank; freeze science and inspect")
    selected = min(eligible, key=lambda record: (
        (record["inner_selection_fixed_decoder_task"]["U"] +
         record["inner_selection_fixed_decoder_task"]["W"]) / 2,
        record["round"]))["round"]
    return {"selected_round": selected, "final_bank_sha256": checks[-1]["final_bank_sha256"],
            "final_bank_checks": checks,
            "selection_rule": "minimum inner-selection balanced task CE among final-bank-feasible Q; earlier round tie"}


def run_center_from_roles(anchor, delta, role_dict, q_ref, historical_q,
                          encoder_sha256, output_dir, max_rounds,
                          *, target_roles=PROTECTED_ROLES,
                          initial_sources=("H", "D17", "Q", "coverage")):
    """Fit a bounded Branch A center from explicit, disjoint development roles.

    Round zero is a complete reference-calibrated, callable 17-token release.
    Each later round adds best-response attack cuts, rebases every reference,
    retains previous decoders and re-solves. Inner checking is recorded but
    never used to choose the round. Independent common audits run separately.
    """
    if not isinstance(anchor, int) or anchor not in (0, 1, 2):
        raise ValueError("anchor must be 0, 1, or 2")
    delta = float(delta)
    if delta not in (0.0, 0.001, 0.003):
        raise ValueError("training delta must be one of the registered allowances")
    if not isinstance(max_rounds, int) or not 0 <= max_rounds <= MAX_ROUNDS:
        raise ValueError("outer decoder/attack round limit outside registration")
    if set(role_dict) != set(SCIENCE_ROLES):
        raise ValueError("fit receives exactly five inner roles and no outer assessment")
    inherited_audit.assert_household_disjoint(
        *(np.asarray(role_dict[name]["households"]) for name in SCIENCE_ROLES))
    q_ref = method.validate_channel(q_ref, n_states=32, n_tokens=17)
    historical_q = method.validate_channel(historical_q, n_states=32, n_tokens=17)
    if not isinstance(encoder_sha256, str) or len(encoder_sha256) != 64:
        raise ValueError("pinned encoder SHA-256 required")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted objects require a private unit directory")
    sources_allowed = {"H", "D17", "Q", "coverage"}
    if (not initial_sources or len(set(initial_sources)) != len(initial_sources)
            or any(source not in sources_allowed for source in initial_sources)
            or "H" not in initial_sources or "D17" not in initial_sources):
        raise ValueError("initial bank requires distinct H/D17 plus registered sources")
    expected = {"schema": "pcrl-adaptive-A-center-v1", "anchor": anchor,
                "delta": delta, "max_rounds": max_rounds,
                "target_roles": list(target_roles),
                "initial_sources": list(initial_sources),
                "encoder_sha256": encoder_sha256,
                "q_ref_sha256": _sha_array(q_ref),
                "historical_q_sha256": _sha_array(historical_q),
                "role_input_sha256": {name: _role_fingerprint(role_dict[name])
                                      for name in SCIENCE_ROLES},
                "outer_labels_accessed": False}
    complete = _load_complete(root, expected)
    if complete is not None:
        return complete
    root.mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(root / "INPUTS.json", expected)
    initial_map = {"H": None, "D17": q_ref, "Q": historical_q,
                   "coverage": (q_ref + historical_q + 1/17) / 3}
    current_q = historical_q
    retained_decoders = {}
    retained_decoder_specs = {}
    retained_cuts = []
    retained_specs = []
    round_records = []
    round_channels = []
    final_calibrated_cuts = None
    final_cost = None
    for round_index in range(max_rounds + 1):
        round_root = root / f"round_r{round_index:02d}"
        seed = 20260924 + 10000 * anchor + 100 * round_index
        decoder_record = fit_frozen_decoder(
            role_dict, current_q, q_ref, root / f"decoder_r{round_index:02d}", seed)
        decoder_id = f"round_{round_index:02d}/{decoder_record['selected_candidate']}"
        retained_decoders[decoder_id] = decoder_record["decoder"]
        retained_decoder_specs[decoder_id] = decoder_record
        sources = ({name: initial_map[name] for name in initial_sources}
                   if round_index == 0 else {f"candidate_r{round_index-1:02d}": current_q})
        fresh_bank = build_attack_bank(
            role_dict, sources, root / f"bank_r{round_index:02d}", seed + 50000,
            target_roles=target_roles, n_states=32)
        retained_cuts.extend(fresh_bank["cuts"])
        retained_specs.extend(fresh_bank["attack_specs"])
        selection_rows = role_dict["inner_selection"]
        valid = _valid_mask(selection_rows, "same_residence")
        chosen = alternate.select_frozen_decoder(
            retained_decoders, np.asarray(selection_rows["ha"])[valid],
            np.asarray(selection_rows["token_codes"])[valid],
            np.asarray(selection_rows["labels"]["same_residence"])[valid],
            np.asarray(selection_rows["weights"])[valid], current_q)
        task_pair = task_cost_pair(chosen["decoder"], role_dict["coefficient_split"])
        solution_record = alternate.channel_update(task_pair, q_ref,
                                                   retained_cuts, delta)
        solution = solution_record["solution"]
        if not solution.get("feasible") or solution.get("Q") is None:
            raise RuntimeError("calibrated center solver unresolved; preserve private unit for diagnosis")
        q = solution["Q"]
        cost_archive = round_root / "COSTS.npz"
        _save_or_verify_npz(cost_archive, cost_U=task_pair["U"],
                            cost_W=task_pair["W"], cost=0.5*(task_pair["U"]+task_pair["W"]))
        cuts_archive = round_root / "CALIBRATED_COEFFICIENTS.npz"
        _save_or_verify_npz(cuts_archive,
                            **{f"cut_{i:04d}": cut["coeff"]
                               for i, cut in enumerate(solution_record["cuts"])})
        _write_or_verify_json(round_root / "CALIBRATED_BANK.json", {
            "schema": 1, "bank_sha256": solution_record["bank_sha256"],
            "coefficients_sha256": _sha_file(cuts_archive),
            "cuts": [{key: value for key, value in cut.items() if key != "coeff"}
                     for cut in solution_record["cuts"]],
            "rho": solution_record["rho"],
            "retained_attack_count": len(retained_specs)})
        channel_root = round_root / "channel"
        artifact = release.ChannelArtifact(
            q, encoder_sha256,
            f"pcrl_adaptive_A_anchor{anchor}_delta{delta:g}_round{round_index}",
            "EXPERIMENTAL_UNVALIDATED")
        if channel_root.exists():
            previous = release.ChannelArtifact.load(channel_root)
            if not np.array_equal(previous.Q, q):
                raise ValueError("existing round channel differs from deterministic replay")
        else:
            artifact.save(channel_root)
        selection_score = task_scores(chosen["decoder"], selection_rows, q)
        check_score = task_scores(chosen["decoder"], role_dict["inner_check"], q)
        record = {"round": round_index, "status": "FIXED_BANK_FEASIBLE",
                  "attack_count": len(retained_specs), "cut_count": len(retained_cuts),
                  "new_attack_count": len(fresh_bank["attack_specs"]),
                  "bank_sha256": solution_record["bank_sha256"],
                  "reference_max_cut_violation": solution_record["witness"]["maximum_cut_violation"],
                  "maximum_cut_violation": solution["replay"]["maximum_cut_violation"],
                  "solver_status": solution["status"],
                  "fixed_decoder_objective": solution["objective"],
                  "fixed_bank_dual_lower_bound": solution["dual_lower_bound"],
                  "fixed_bank_gap": solution["fixed_bank_gap"],
                  "selected_decoder_id": chosen["id"],
                  "selected_decoder_model_sha256":
                      retained_decoder_specs[chosen["id"]]["selected_model_sha256"],
                  "inner_selection_fixed_decoder_task": selection_score,
                  "inner_check_fixed_decoder_task": check_score,
                  "task_cost_archive_sha256": _sha_file(cost_archive),
                  "calibrated_bank_manifest_sha256": _sha_file(round_root / "CALIBRATED_BANK.json"),
                  "channel_file_sha256": _sha_file(channel_root / "Q.npz"),
                  "outer_labels_accessed": False,
                  "scope": "fixed-bank/fixed-decoder training and inner task diagnostics; independent common audit is separate"}
        _write_or_verify_json(round_root / "ROUND.json", record)
        round_records.append(record)
        round_channels.append(q.copy())
        final_calibrated_cuts = solution_record["cuts"]
        final_cost = 0.5*(task_pair["U"]+task_pair["W"])
        current_q = q
    final_selection = select_final_bank_feasible(
        round_channels, round_records, final_cost, final_calibrated_cuts)
    _write_or_verify_json(root / "FINAL_BANK_SELECTION.json", final_selection)
    selected_round = final_selection["selected_round"]
    _write_or_verify_json(root / "SELECTED.json", {
        "schema": 1, "selected_round": selected_round,
        "selection_rule": final_selection["selection_rule"],
        "final_bank_sha256": final_selection["final_bank_sha256"],
        "final_bank_selection_sha256": _sha_file(root / "FINAL_BANK_SELECTION.json"),
        "channel_relative_directory": f"round_r{selected_round:02d}/channel",
        "channel_sha256": round_records[selected_round]["channel_file_sha256"],
        "independent_common_audit_completed": False})
    receipt = {**expected, "status": "COMPLETE", "selected_round": selected_round,
               "rounds": round_records, "artifact_sha256": _inventory(root)}
    _write_or_verify_json(root / "COMPLETE.json", receipt)
    return _load_complete(root, expected)


def run_center(anchor, delta, index_path, output_dir, max_rounds):
    """Hash-verified 2018 Branch A fit; requires sanitized outer-label barrier."""
    sanitized = (Path(__file__).resolve().parents[2] /
                 data.SANITIZED_RELATIVE / "SANITIZATION.json")
    if not sanitized.is_file():
        raise RuntimeError("sanitized 2018 receipt required; refusing raw prepared fallback")
    value = data.index(index_path)
    prepared = data.load_prepared(value, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise RuntimeError("outer assessment labels appeared before lock")
    role_dict = {name: roles.pooled_role(prepared, name) for name in SCIENCE_ROLES}
    q_ref = data.load_map(value, anchor, "D17")
    historical_q = data.load_map(value, anchor, "Q")
    encoder_sha = data.member_record(value, anchor, "encoder")["sha256"]
    return run_center_from_roles(anchor, delta, role_dict, q_ref, historical_q,
                                 encoder_sha, output_dir, max_rounds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--delta", type=float, required=True, choices=(0., .001, .003))
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rounds", type=int, default=0)
    args = parser.parse_args(argv)
    record = run_center(args.anchor, args.delta, args.index,
                        args.output_dir, args.max_rounds)
    print(json.dumps({"status": record["status"], "selected_round": record["selected_round"],
                      "complete_receipt_sha256": record["complete_receipt_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
