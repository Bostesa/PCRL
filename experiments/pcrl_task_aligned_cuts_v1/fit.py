"""First 2018 development U1/P1 fit from hash-verified frozen objects.

This module never reads attacker_validation or a 2016 person row. Predictor
training uses attacker_fit or downstream_fit, selection uses the registered
inner half of downstream_validation, and coefficients use the historical
representation_fit/mechanism rows. Outputs must be in a private directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from . import audit, data, method, release, solver

ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")
SOURCES = ("H", "D17", "Q", "coverage")
CLASS_COUNT = {"SEX": 2, "RAC1P": 9}


def _private_output(directory, *, resume=False):
    path = Path(directory)
    if "private" not in path.parts:
        raise ValueError("fitted predictors, cuts and person-derived objects require a private output path")
    if path.exists():
        if any(path.iterdir()) and not resume:
            raise FileExistsError("private scientific unit already has artifacts; never overwrite")
    else:
        path.mkdir(parents=True)
    return path


def _save_or_verify_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays) or any(
                    not np.array_equal(stored[key], np.asarray(value))
                    for key, value in arrays.items()):
                raise ValueError(f"existing private archive differs from recomputation: {path}")
    else:
        np.savez_compressed(path, **arrays)


def _write_or_verify_json(path, value):
    path = Path(path)
    converted = _jsonable(value)
    if path.exists():
        if json.loads(path.read_text()) != converted:
            raise ValueError(f"existing private JSON differs from recomputation: {path}")
    else:
        _write_json(path, converted)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _selection_rows(value, anchor, provided):
    """Use one registered household split; a supplied mask is checked exactly."""
    all_prepared = {k: data.load_prepared(value, k) for k in (0, 1, 2)}
    split = data.global_validation_split(all_prepared)
    expected = np.asarray(split["anchors"][anchor]["rows"]["inner_selection"], dtype=np.int64)
    if provided is not None:
        rows = np.asarray(provided)
        if rows.ndim != 1:
            raise ValueError("inner selection rows must be one-dimensional")
        if rows.dtype.kind == "b":
            rows = np.flatnonzero(rows)
        elif rows.dtype.kind not in "iu":
            raise ValueError("inner selection rows must be bool or integer indices")
        if not np.array_equal(np.asarray(rows, dtype=np.int64), expected):
            raise ValueError("inner selection does not match frozen global household split")
    return all_prepared[anchor], expected, split["assignment_sha256"]


def _valid_task_rows(pool):
    y = np.asarray(pool["labels"]["same_residence"])
    mask = np.isin(y, [0, 1])
    if not np.any(mask):
        raise ValueError("no valid residence labels on declared training resource")
    return mask, int((~mask).sum())


def _codes(prepared, pool_name, rows=None):
    t = np.asarray(prepared["encoded"][pool_name]["codes"]["T0"], dtype=np.int64)
    return t if rows is None else t[rows]


def _model_hash(directory):
    path = Path(directory)
    files = sorted(p for p in path.iterdir() if p.is_file())
    if not files:
        raise ValueError("empty fitted model directory")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(bytes.fromhex(data.sha256_file(file)))
    return digest.hexdigest()


def _coefficient_pool_hash(ids, t, labels, weights):
    digest = hashlib.sha256()
    for value in ids:
        digest.update(hashlib.sha256(str(value).encode()).digest())
    for a in (t, labels, weights):
        a = np.ascontiguousarray(a)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _source_q(source, q, d17):
    if source == "H":
        return np.ones((len(q), 1), dtype=np.float64)
    if source == "Q":
        return q
    if source == "D17":
        return d17
    if source == "coverage":
        return (q + d17 + 1/17)/3
    raise ValueError("unregistered attack source")


def fit_u1(anchor, index_path, output_dir, inner_selection_mask=None, *, seed=None):
    """Fit one shared U1 decoder; persist its selected private checkpoint.

    The coefficient pool and outer attacker_validation are not consulted for
    model selection. All three anchors are loaded solely to reproduce the
    precommitted household assignment.
    """
    value = data.index(index_path)
    prepared, selection, split_sha = _selection_rows(value, anchor, inner_selection_mask)
    historical_q, d17 = (data.load_map(value, anchor, name) for name in ("Q", "D17"))
    train = prepared["ctx"]["pools"]["downstream_fit"]
    validation = prepared["ctx"]["pools"]["downstream_validation"]
    audit.assert_household_disjoint(train["households"], validation["households"][selection])
    fit_mask, fit_excluded = _valid_task_rows(train)
    select_y = np.asarray(validation["labels"]["same_residence"])[selection]
    select_valid = np.isin(select_y, [0, 1])
    if not np.any(select_valid):
        raise ValueError("no valid residence labels in inner selection")
    select = selection[select_valid]
    root = _private_output(output_dir)
    chosen = method.fit_common_decoder(
        train["ha"][fit_mask], _codes(prepared, "downstream_fit")[fit_mask],
        np.asarray(train["labels"]["same_residence"])[fit_mask], train["weights"][fit_mask],
        validation["ha"][select], _codes(prepared, "downstream_validation")[select],
        np.asarray(validation["labels"]["same_residence"])[select], validation["weights"][select],
        historical_q, d17, seed=int(20260923 + 100*anchor if seed is None else seed),
        out_dir=root / "decoder_slate")
    selected_dir = root / "decoder_slate" / chosen["selection"]
    record = {"schema": 1, "anchor": anchor, "selection": chosen["selection"],
              "selected_model_directory": str(selected_dir.resolve()),
              "selected_model_relative": str(selected_dir.relative_to(root)),
              "selected_model_sha256": _model_hash(selected_dir),
              "slate_sha256": data.sha256_file(root / "decoder_slate" / "slate.json"),
              "selection_household_assignment_sha256": split_sha,
              "fit_original_people": int(fit_mask.sum()),
              "fit_missing_task_excluded": fit_excluded,
              "validation_original_people": int(len(select)),
              "validation_missing_task_excluded": int((~select_valid).sum()),
              "training_law": chosen["training_law"],
              "selection_rule": "minimum balanced U/W validation expected-token log loss, lexical tie",
              "frozen_before_channel_fit": True,
              "created_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(root / "FIT_U1.json", record)
    return record


def _selected_reference(routes, score_key="balanced"):
    score_key = {"U": "unweighted", "W": "weighted"}.get(score_key, score_key)
    if score_key not in ("balanced", "unweighted", "weighted"):
        raise ValueError("reference score must be a declared inner-selection weighting")
    choices = []
    for source_view, source, registry in routes:
        if source not in ("H", "D17"):
            continue
        for cid in registry["own_scores"]:
            if "models" in registry and cid not in registry["models"]:
                raise ValueError("reference score has no saved fitted predictor")
            score = float(registry["own_scores"][cid][score_key])
            choices.append((score, source_view, source, cid, registry))
    _, source_view, source, cid, registry = min(choices, key=lambda value: value[:4])
    return source_view, source, cid, registry


def _candidate_coefficients(model_directory, source, source_view, role,
                            coefficient_rows, coefficient_codes, valid_mask):
    from experiments.pcrl_task_directed_release_v1.audits import load_candidate
    candidate = load_candidate(model_directory)
    _, target = role.split("/")
    h = audit.view_features(coefficient_rows["ha"], coefficient_rows["hb"], source_view)
    if candidate.n_classes != CLASS_COUNT[target]:
        raise ValueError("attack candidate changed full class schema")
    if source == "H":
        if candidate.n_tokens != 1:
            raise ValueError("H-only ancestor must ignore token")
        prediction = np.repeat(candidate.predict_token_proba(h, 1), 17, axis=1)
    else:
        if candidate.n_tokens != 17:
            raise ValueError("source token alphabet differs from common 17 actions")
        prediction = candidate.predict_token_proba(h, 17)
    labels = np.asarray(coefficient_rows["labels"][target])[valid_mask]
    return method.coefficient_pair(coefficient_codes[valid_mask], labels,
                                   prediction[valid_mask], 32,
                                   coefficient_rows["weights"][valid_mask])


def fit_p1_center(anchor, index_path, u1_dir, output_dir, selection_mask=None,
                  *, source_channels=SOURCES, delta=0.0, seed=None,
                  attack_slate="standard", resume=False):
    """Fit initial full-view attack bank and anchor P1 center channel.

    A source subset is an explicitly incomplete engineering checkpoint. This
    function fits no best-response exchange attacks and opens no outer data.
    """
    sources = tuple(source_channels)
    if not sources or len(set(sources)) != len(sources) or any(s not in SOURCES for s in sources):
        raise ValueError("source_channels must be unique registered sources")
    if "H" not in sources or "D17" not in sources:
        raise ValueError("H-only and D17 are mandatory reference sources")
    if attack_slate not in ("standard", "catchup"):
        raise ValueError("unregistered attack slate")
    delta = float(delta)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    value = data.index(index_path)
    prepared, selection, split_sha = _selection_rows(value, anchor, selection_mask)
    q_hist, d17 = (data.load_map(value, anchor, name) for name in ("Q", "D17"))
    coefficient_rows = data.coefficient_pool(prepared, d17)
    mechanism_indices = np.asarray(prepared["roles"]["mechanism"], dtype=np.int64)
    coefficient_codes = _codes(prepared, "representation_fit", mechanism_indices)
    task_mask, task_excluded = _valid_task_rows(coefficient_rows)
    if u1_dir is None:
        # Historical fixed action decoder, identical coefficient rows and
        # clipping; this permits bank generation before the U1 fit completes.
        frozen = prepared["tables"]["T0"]
        task_pair = {"U": np.asarray(frozen["cost_U"], dtype=np.float64),
                     "W": np.asarray(frozen["cost_W"], dtype=np.float64)}
        u1_sha = None
        arm = "U0P1"
    else:
        u1 = json.loads((Path(u1_dir) / "FIT_U1.json").read_text())
        if u1["anchor"] != anchor or u1["selection_household_assignment_sha256"] != split_sha:
            raise ValueError("U1 decoder anchor/inner selection mismatch")
        decoder_dir = Path(u1_dir) / u1["selected_model_relative"]
        if _model_hash(decoder_dir) != u1["selected_model_sha256"]:
            raise ValueError("U1 decoder object hash mismatch")
        from experiments.pcrl_task_directed_release_v1.audits import load_candidate
        decoder = load_candidate(decoder_dir)
        task_pair = method.task_cost_from_decoder(
            decoder, coefficient_rows["ha"][task_mask], coefficient_codes[task_mask],
            np.asarray(coefficient_rows["labels"]["same_residence"])[task_mask],
            coefficient_rows["weights"][task_mask])
        u1_sha = u1["selected_model_sha256"]
        arm = "U1P1"
    cost = .5*(task_pair["U"]+task_pair["W"])
    d_u1 = method.rowwise_unconstrained(cost)
    root = _private_output(output_dir, resume=resume)
    if resume and (root / "FIT_P1_CENTER.json").exists():
        old = json.loads((root / "FIT_P1_CENTER.json").read_text())
        if (old.get("anchor") != anchor or old.get("arm") != arm or
                old.get("delta") != delta or
                old.get("selection_household_assignment_sha256") != split_sha or
                data.sha256_file(root / "coefficients.npz") != old.get("cost_archive_sha256") or
                json.loads((root / "bank.json").read_text()).get("bank_sha256") != old.get("bank_sha256") or
                (old.get("channel_path") is not None and
                 data.sha256_file(root / "channel" / "Q.npz") != old.get("channel_sha256"))):
            raise ValueError("existing completed center does not match requested scientific unit")
        return old
    _save_or_verify_npz(root / "UNCONSTRAINED.npz", Q=d_u1["Q"])
    attack_root = root / "attack_bank"
    attack_root.mkdir(exist_ok=resume)
    registries = {}
    b_registries = {}
    input_support = {}
    initial_cuts = []
    fit_codes = _codes(prepared, "attacker_fit")
    validation_codes = _codes(prepared, "downstream_validation", selection)
    base_seed = int(20261000 + 1000*anchor if seed is None else seed)
    for role_index, role in enumerate(ROLES):
        view, target = role.split("/")
        target_labels = np.asarray(coefficient_rows["labels"][target])
        valid = np.isfinite(target_labels) & (target_labels == np.floor(target_labels)) & (
            target_labels >= 0) & (target_labels < CLASS_COUNT[target])
        if not np.any(valid):
            raise ValueError("protected coefficient role has no supported labels")
        counts = np.bincount(target_labels[valid].astype(int), minlength=CLASS_COUNT[target])
        input_support[role] = {"coefficient_original_people": int(valid.sum()),
                               "coefficient_missing_excluded": int((~valid).sum()),
                               "coefficient_class_counts": counts.tolist(),
                               "coefficient_missing_classes": np.flatnonzero(counts == 0).tolist()}
        role_registries = {}
        for source_index, source in enumerate(sources):
            source_q = _source_q(source, q_hist, d17)
            fit_rows = data.token_pool(prepared, "attacker_fit", source_q)
            validation_full = data.token_pool(prepared, "downstream_validation", source_q)
            validation_rows = data.subset_token_pool(validation_full, selection)
            fit_dir = attack_root / role.replace("/", "_") / source
            role_registries[source] = audit.fit_role_slate(
                fit_rows, fit_rows["token_probs"], validation_rows,
                validation_rows["token_probs"], f"attack:{role}", fit_dir,
                base_seed + 100*role_index + source_index, release_id=source,
                slate=attack_slate)
        registries[role] = role_registries
        routes = [(view, source, role_registries[source]) for source in sources]
        if view == "AB":
            # A can ignore H_B and run any same-alphabet A token predictor;
            # B can ignore the token and run its H_B-only predictor. These are
            # legally available coalition ancestors, not extra released data.
            routes.extend(("A", source, registries[f"A/{target}"][source])
                          for source in sources)
            if target not in b_registries:
                h_q = np.ones((32, 1), dtype=np.float64)
                b_fit = data.token_pool(prepared, "attacker_fit", h_q)
                b_validation = data.subset_token_pool(
                    data.token_pool(prepared, "downstream_validation", h_q), selection)
                b_registries[target] = audit.fit_role_slate(
                    b_fit, b_fit["token_probs"], b_validation,
                    b_validation["token_probs"], f"attack:B/{target}",
                    attack_root / f"B_{target}" / "H",
                    base_seed + 1000 + role_index, release_id="H",
                    slate=attack_slate)
            routes.append(("B", "H", b_registries[target]))
        references = {}
        for weighting, score_key in (("U", "unweighted"), ("W", "weighted")):
            ref_view, ref_source, ref_cid, ref_registry = _selected_reference(routes, score_key)
            ref_model_dir = ref_registry["models"][ref_cid]["model_directory"]
            ref_pair = _candidate_coefficients(ref_model_dir, ref_source, ref_view,
                                               role, coefficient_rows, coefficient_codes, valid)
            references[weighting] = {
                "rho": float(np.sum(ref_pair[weighting] * d17)),
                "source_view": ref_view, "source": ref_source,
                "candidate": ref_cid, "predictor_sha256": _model_hash(ref_model_dir)}
        for source_view, source, registry in routes:
            for cid, model in sorted(registry["models"].items()):
                model_dir = model["model_directory"]
                pair = _candidate_coefficients(model_dir, source, source_view, role,
                                               coefficient_rows, coefficient_codes, valid)
                predictor_sha = _model_hash(model_dir)
                for weighting in ("U", "W"):
                    reference = references[weighting]
                    rho = reference["rho"]
                    floor = rho - delta
                    initial_cuts.append({
                        "id": f"{role}/from_{source_view}/{source}/{cid}/{weighting}",
                        "coeff": pair[weighting], "floor": floor,
                        "rho": rho, "delta": delta,
                        "view": view, "target": target,
                        "class_order": list(range(CLASS_COUNT[target])),
                        "weighting": weighting,
                        "predictor_sha256": predictor_sha,
                        "coefficient_pool_sha256": _coefficient_pool_hash(
                            np.asarray(coefficient_rows["ids"])[valid],
                            coefficient_codes[valid], target_labels[valid],
                            coefficient_rows["weights"][valid]),
                        "weight_normalization": "1/n" if weighting == "U" else "PWGTP/sum(PWGTP)",
                        "reference_source": reference["source"],
                        "reference_source_view": reference["source_view"],
                        "reference_candidate": reference["candidate"],
                        "reference_selection_weighting": weighting,
                        "reference_predictor_sha256": reference["predictor_sha256"],
                        "source_release": source,
                        "source_view": source_view,
                        "model_candidate": cid})
    # Keep exact grouped empirical coefficients private for deterministic and
    # gradient controls; expose only hashes and aggregate replay publicly.
    _save_or_verify_npz(root / "coefficients.npz", cost_U=task_pair["U"],
                        cost_W=task_pair["W"], cost=cost,
                        **{f"cut_{i:04d}": cut["coeff"] for i, cut in enumerate(initial_cuts)})
    _write_or_verify_json(root / "bank.json", {
        "bank_sha256": solver.bank_sha256(initial_cuts, cost.shape),
        "cuts": [{k: v for k, v in cut.items() if k != "coeff"} for cut in initial_cuts],
        "coefficient_archive_sha256": data.sha256_file(root / "coefficients.npz")})
    solution = solver.solve_p1(cost, initial_cuts)
    registered_feasible = bool(solution["feasible"])
    fallback_relaxation = None
    if not registered_feasible and solution["status"] == "registered_bank_infeasible":
        fallback_relaxation = float(solution["phase_one"]["minimum_common_violation"] + solver.PRIMAL_TOL)
        fallback = solver.relax_cuts(initial_cuts, fallback_relaxation, cost.shape)
        solution = solver.solve_p1(cost, fallback)
    channel_path = None
    if solution.get("Q") is not None and solution.get("feasible"):
        encoder_sha = data.member_record(value, anchor, "encoder")["sha256"]
        channel_path = str((root / "channel").resolve())
        artifact = release.ChannelArtifact(solution["Q"], encoder_sha,
                                           f"{arm}_delta{delta:+g}_initial_bank_anchor_{anchor}",
                                           "EXPERIMENTAL_UNVALIDATED")
        if (root / "channel").exists():
            existing = release.ChannelArtifact.load(root / "channel")
            if (not np.array_equal(existing.Q, artifact.Q) or
                    existing.encoder_sha256 != artifact.encoder_sha256 or
                    existing.source != artifact.source):
                raise ValueError("existing channel differs from resumed fixed-bank solve")
        else:
            artifact.save(channel_path)
    record = {"schema": 1, "anchor": anchor, "arm": arm, "delta": delta,
              "attack_slate": attack_slate,
              "source_channels": list(sources), "bank_complete": set(sources) == set(SOURCES),
              "exchange_rounds": 0,
              "status": "INITIAL_BANK_FEASIBLE" if registered_feasible else
                        ("DESCRIPTIVE_PHASE_I_FALLBACK" if channel_path else "NO_CHANNEL"),
              "registered_feasible": registered_feasible,
              "fallback_common_relaxation": fallback_relaxation,
              "channel_path": channel_path,
              "channel_relative": "channel" if channel_path else None,
              "channel_sha256": data.sha256_file(Path(channel_path) / "Q.npz") if channel_path else None,
              "cost_archive_sha256": data.sha256_file(root / "coefficients.npz"),
              "bank_sha256": solver.bank_sha256(initial_cuts, cost.shape),
              "n_cuts": len(initial_cuts),
              "unconstrained_reference": "D_U1" if arm == "U1P1" else "D17",
              "unconstrained_objective": d_u1["objective"],
              "unconstrained_channel_sha256": data.sha256_file(root / "UNCONSTRAINED.npz"),
              "task_missing_excluded": task_excluded,
              "role_support": input_support,
              "selection_household_assignment_sha256": split_sha,
              "u1_selected_model_sha256": u1_sha,
              "solution": {k: v for k, v in solution.items() if k != "Q"},
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "initial fitted bank only; no best-response exchange or independent pilot audit"}
    _write_json(root / "FIT_P1_CENTER.json", record)
    return record


def fit_reference_bank(anchor, index_path, output_dir, selection_mask=None,
                       *, source_channels=SOURCES, seed=None,
                       attack_slate="standard", resume=False):
    """Fit the cost-independent initial attack bank once per anchor.

    The bank-generation run also solves its U0P1/delta=0 fixed-bank channel,
    which is a registered crossed arm. U1 and other deltas reuse these attacks
    and coefficient rows rather than refitting them opportunistically.
    """
    return fit_p1_center(anchor, index_path, None, output_dir, selection_mask,
                         source_channels=source_channels, delta=0., seed=seed,
                         attack_slate=attack_slate, resume=resume)


def exchange_round(anchor, index_path, bank_dir, cost_dir, channels, output_dir,
                   *, round_index, delta=0., selection_mask=None,
                   attack_slate="standard", resume=False):
    """One shared-bank best-response round on inner development resources.

    ``channels`` maps fixed candidate names (normally LP and deterministic)
    to 32-by-17 kernels. The caller must run the matched deterministic solver
    on the resulting frozen bank before the next round. The round fits no
    final-audit or outer-assessment predictor.
    """
    from . import controls
    if not 1 <= int(round_index) <= 12 or int(round_index) != round_index:
        raise ValueError("exchange round is outside the registered maximum")
    if attack_slate not in ("standard", "catchup"):
        raise ValueError("unregistered response slate")
    if not channels or "LP" not in channels:
        raise ValueError("exchange requires the LP candidate and any matched controls")
    value = data.index(index_path)
    prepared, selection, split_sha = _selection_rows(value, anchor, selection_mask)
    _, old_cuts, old_bank_sha = controls.load_saved_bank(bank_dir)
    cost_path = Path(cost_dir) / "cost.npz"
    if not cost_path.exists():
        cost_path = Path(cost_dir) / "coefficients.npz"
    with np.load(cost_path, allow_pickle=False) as saved_cost:
        cost_u = np.asarray(saved_cost["cost_U"], dtype=np.float64)
        cost_w = np.asarray(saved_cost["cost_W"], dtype=np.float64)
        cost = np.asarray(saved_cost["cost"], dtype=np.float64)
    if cost.shape != (32, 17) or not np.allclose(cost, .5*(cost_u+cost_w), atol=1e-13):
        raise ValueError("frozen task cost scaling differs from registered objective")
    if any(abs(float(cut["rho"])-float(delta)-float(cut["floor"])) > 1e-10
           for cut in old_cuts):
        raise ValueError("old bank uses another registered protection delta")
    kernels = {name: release.validate_channel(np.asarray(q), n_states=32, n_tokens=17)
               for name, q in channels.items()}
    root = _private_output(output_dir, resume=resume)
    if resume and (root / "EXCHANGE_ROUND.json").exists():
        old = json.loads((root / "EXCHANGE_ROUND.json").read_text())
        if (old["anchor"] != anchor or old["round_index"] != round_index or
                old["source_bank_sha256"] != old_bank_sha or
                old["selection_household_assignment_sha256"] != split_sha or
                data.sha256_file(root / "coefficients.npz") != old["coefficient_archive_sha256"]):
            raise ValueError("completed exchange round does not match frozen inputs")
        return old
    coefficient_rows = data.coefficient_pool(prepared, data.load_map(value, anchor, "D17"))
    mechanism = np.asarray(prepared["roles"]["mechanism"], dtype=np.int64)
    coefficient_codes = _codes(prepared, "representation_fit", mechanism)
    references = {}
    for cut in old_cuts:
        role, weighting = cut["view"]+"/"+cut["target"], cut["weighting"]
        key = (role, weighting)
        if key in references and abs(references[key]["rho"]-cut["rho"]) > 1e-12:
            raise ValueError("bank changed its frozen reference risk")
        references[key] = {k: cut[k] for k in (
            "rho", "reference_source", "reference_source_view",
            "reference_candidate", "reference_predictor_sha256",
            "coefficient_pool_sha256", "class_order", "weight_normalization")}
    if set(references) != {(role, weight) for role in ROLES for weight in ("U", "W")}:
        raise ValueError("old bank lacks a protected role/weighting reference")
    # Exact aliases share one response fit, but all declared labels remain in
    # the provenance map. A changed channel is a new release and new bank fit.
    unique = {}
    aliases = {}
    for name, q in sorted(kernels.items()):
        digest = hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()
        aliases[name] = digest
        unique.setdefault(digest, (name, q))
    proposals = []
    for digest, (name, q) in unique.items():
        registries = {}
        fit_rows = data.token_pool(prepared, "attacker_fit", q)
        validation_rows = data.subset_token_pool(
            data.token_pool(prepared, "downstream_validation", q), selection)
        for role_index, role in enumerate(ROLES):
            view, target = role.split("/")
            registry = audit.fit_role_slate(
                fit_rows, fit_rows["token_probs"], validation_rows,
                validation_rows["token_probs"], f"attack:{role}",
                root / "response_slates" / digest / role.replace("/", "_"),
                20262000 + 10000*anchor + 100*int(round_index) + role_index,
                release_id=f"exchange_r{round_index}_{digest[:16]}", slate=attack_slate)
            registries[role] = registry
            routes = [(view, registry)]
            if view == "AB":
                routes.append(("A", registries[f"A/{target}"]))
            target_labels = np.asarray(coefficient_rows["labels"][target])
            valid = np.isfinite(target_labels) & (target_labels == np.floor(target_labels)) & (
                target_labels >= 0) & (target_labels < CLASS_COUNT[target])
            for source_view, route_registry in routes:
                for cid, model in sorted(route_registry["models"].items()):
                    model_dir = model["model_directory"]
                    pair = _candidate_coefficients(model_dir, name, source_view, role,
                                                   coefficient_rows, coefficient_codes, valid)
                    for weighting in ("U", "W"):
                        ref = references[(role, weighting)]
                        proposals.append({
                            "id": f"exchange_r{round_index}/{digest[:16]}/{role}/from_{source_view}/{cid}/{weighting}",
                            "coeff": pair[weighting], "floor": float(ref["rho"])-float(delta),
                            "rho": ref["rho"], "delta": float(delta),
                            "view": view, "target": target, "weighting": weighting,
                            "class_order": ref["class_order"],
                            "weight_normalization": ref["weight_normalization"],
                            "coefficient_pool_sha256": ref["coefficient_pool_sha256"],
                            "reference_source": ref["reference_source"],
                            "reference_source_view": ref["reference_source_view"],
                            "reference_candidate": ref["reference_candidate"],
                            "reference_predictor_sha256": ref["reference_predictor_sha256"],
                            "source_release": name, "source_channel_sha256": digest,
                            "source_view": source_view, "model_candidate": cid,
                            "predictor_sha256": _model_hash(model_dir)})
    union = solver.add_violated_cuts(old_cuts, proposals, kernels, cost.shape)
    _save_or_verify_npz(root / "coefficients.npz", cost_U=cost_u, cost_W=cost_w,
                        cost=cost, **{f"cut_{i:04d}": cut["coeff"]
                                     for i, cut in enumerate(union["cuts"])})
    archive_sha = data.sha256_file(root / "coefficients.npz")
    _write_or_verify_json(root / "bank.json", {
        "bank_sha256": union["bank_sha256"],
        "cuts": [{k: v for k, v in cut.items() if k != "coeff"} for cut in union["cuts"]],
        "coefficient_archive_sha256": archive_sha})
    solution = solver.solve_p1(cost, union["cuts"])
    channel_relative = None
    if solution.get("Q") is not None and solution.get("feasible"):
        channel_relative = "channel"
        release.ChannelArtifact(solution["Q"],
            data.member_record(value, anchor, "encoder")["sha256"],
            f"P1_exchange_r{round_index}_anchor_{anchor}",
            "EXPERIMENTAL_UNVALIDATED").save(root / channel_relative)
    record = {"schema": 1, "anchor": anchor, "round_index": int(round_index),
              "delta": float(delta), "attack_slate": attack_slate,
              "source_bank_sha256": old_bank_sha,
              "bank_sha256": union["bank_sha256"],
              "coefficient_archive_sha256": archive_sha,
              "new_cut_count": len(union["added_ids"]),
              "total_cut_count": len(union["cuts"]),
              "oracle_log": union["oracle_log"],
              "response_channel_aliases_sha256": aliases,
              "synchronized_lp_deterministic": "LP" in kernels and "DET" in kernels,
              "channel_relative": channel_relative,
              "channel_sha256": data.sha256_file(root / channel_relative / "Q.npz")
              if channel_relative else None,
              "solution": {k: v for k, v in solution.items() if k != "Q"},
              "selection_household_assignment_sha256": split_sha,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "one heuristic best-response exchange round; re-solve matched deterministic control on this exact bank"}
    _write_json(root / "EXCHANGE_ROUND.json", record)
    return record


def solve_from_bank(anchor, index_path, bank_dir, output_dir, *, arm="U1P1",
                    delta=0., u1_dir=None, selection_mask=None):
    """Solve a crossed P1 arm against the frozen initial attack bank."""
    if arm not in ("U0P1", "U1P1"):
        raise ValueError("solve_from_bank supports only registered P1 arms")
    if (arm == "U1P1") != (u1_dir is not None):
        raise ValueError("U1P1 requires a frozen U1 decoder; U0P1 uses historical costs")
    delta = float(delta)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    value = data.index(index_path)
    prepared, _, split_sha = _selection_rows(value, anchor, selection_mask)
    bank_root = Path(bank_dir)
    bank_fit = json.loads((bank_root / "FIT_P1_CENTER.json").read_text())
    manifest = json.loads((bank_root / "bank.json").read_text())
    if (bank_fit["anchor"] != anchor or bank_fit["arm"] != "U0P1" or
            bank_fit["selection_household_assignment_sha256"] != split_sha or
            manifest["bank_sha256"] != bank_fit["bank_sha256"] or
            data.sha256_file(bank_root / "coefficients.npz") != manifest["coefficient_archive_sha256"]):
        raise ValueError("frozen attack bank/anchor/selection/hash mismatch")
    with np.load(bank_root / "coefficients.npz", allow_pickle=False) as archive:
        historical_u, historical_w = archive["cost_U"].copy(), archive["cost_W"].copy()
        cuts = []
        for i, metadata in enumerate(manifest["cuts"]):
            coefficient = archive[f"cut_{i:04d}"].copy()
            cut = {**metadata, "coeff": coefficient,
                   "delta": delta, "floor": float(metadata["rho"])-delta}
            cuts.append(cut)
    original_cuts = [{**cut, "delta": 0., "floor": float(cut["rho"])} for cut in cuts]
    if solver.bank_sha256(original_cuts, historical_u.shape) != manifest["bank_sha256"]:
        raise ValueError("bank coefficient replay differs from frozen hash")
    if arm == "U0P1":
        cost_u, cost_w = historical_u, historical_w
        u1_sha = None
    else:
        u1 = json.loads((Path(u1_dir) / "FIT_U1.json").read_text())
        if u1["anchor"] != anchor or u1["selection_household_assignment_sha256"] != split_sha:
            raise ValueError("U1 decoder anchor/selection mismatch")
        decoder_dir = Path(u1_dir) / u1["selected_model_relative"]
        if _model_hash(decoder_dir) != u1["selected_model_sha256"]:
            raise ValueError("U1 model hash mismatch")
        from experiments.pcrl_task_directed_release_v1.audits import load_candidate
        decoder = load_candidate(decoder_dir)
        coefficient_rows = data.coefficient_pool(prepared, data.load_map(value, anchor, "D17"))
        mechanism_indices = np.asarray(prepared["roles"]["mechanism"], dtype=np.int64)
        t = _codes(prepared, "representation_fit", mechanism_indices)
        mask, _ = _valid_task_rows(coefficient_rows)
        pair = method.task_cost_from_decoder(
            decoder, coefficient_rows["ha"][mask], t[mask],
            np.asarray(coefficient_rows["labels"]["same_residence"])[mask],
            coefficient_rows["weights"][mask])
        cost_u, cost_w = pair["U"], pair["W"]
        u1_sha = u1["selected_model_sha256"]
    cost = .5*(cost_u+cost_w)
    root = _private_output(output_dir)
    np.savez_compressed(root / "cost.npz", cost_U=cost_u, cost_W=cost_w, cost=cost)
    unconstrained = method.rowwise_unconstrained(cost)
    np.savez_compressed(root / "UNCONSTRAINED.npz", Q=unconstrained["Q"])
    solution = solver.solve_p1(cost, cuts)
    registered_feasible = bool(solution["feasible"])
    fallback_relaxation = None
    if not registered_feasible and solution["status"] == "registered_bank_infeasible":
        fallback_relaxation = float(solution["phase_one"]["minimum_common_violation"]+solver.PRIMAL_TOL)
        solution = solver.solve_p1(cost, solver.relax_cuts(cuts, fallback_relaxation, cost.shape))
    channel_relative = None
    if solution.get("Q") is not None and solution.get("feasible"):
        encoder_sha = data.member_record(value, anchor, "encoder")["sha256"]
        channel_relative = "channel"
        release.ChannelArtifact(solution["Q"], encoder_sha,
                                f"{arm}_delta{delta:+g}_frozen_bank_anchor_{anchor}",
                                "EXPERIMENTAL_UNVALIDATED").save(root / channel_relative)
    record = {"schema": 1, "anchor": anchor, "arm": arm, "delta": delta,
              "attack_slate": bank_fit["attack_slate"],
              "source_channels": bank_fit["source_channels"],
              "bank_complete": bank_fit["bank_complete"],
              "exchange_rounds": 0, "frozen_source_bank_sha256": manifest["bank_sha256"],
              "adjusted_bank_sha256": solver.bank_sha256(cuts, cost.shape),
              "n_cuts": len(cuts),
              "status": "INITIAL_BANK_FEASIBLE" if registered_feasible else
                        ("DESCRIPTIVE_PHASE_I_FALLBACK" if channel_relative else "NO_CHANNEL"),
              "registered_feasible": registered_feasible,
              "fallback_common_relaxation": fallback_relaxation,
              "channel_relative": channel_relative,
              "channel_sha256": data.sha256_file(root / channel_relative / "Q.npz")
              if channel_relative else None,
              "cost_sha256": data.sha256_file(root / "cost.npz"),
              "unconstrained_reference": "D_U1" if arm == "U1P1" else "D17",
              "unconstrained_objective": unconstrained["objective"],
              "unconstrained_channel_sha256": data.sha256_file(root / "UNCONSTRAINED.npz"),
              "u1_selected_model_sha256": u1_sha,
              "selection_household_assignment_sha256": split_sha,
              "role_support": bank_fit["role_support"],
              "solution": {k: v for k, v in solution.items() if k != "Q"},
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "initial frozen fitted bank only; no best-response exchange or independent pilot audit"}
    _write_json(root / "FIT_P1_ARM.json", record)
    return record


def solve_p0_arm(anchor, index_path, output_dir, *, arm="U0P0",
                 budget=0.01, u1_dir=None, selection_mask=None):
    """Registered A-only historical-CMI arm with U0 or frozen common U1 cost."""
    if arm not in ("U0P0", "U1P0") or (arm == "U1P0") != (u1_dir is not None):
        raise ValueError("P0 arm/U1 decoder mismatch")
    budget = float(budget)
    if budget not in (0.005, 0.01, 0.02):
        raise ValueError("P0 budget is outside registered pilot set")
    value = data.index(index_path)
    prepared, _, split_sha = _selection_rows(value, anchor, selection_mask)
    historical = prepared["tables"]["T0"]
    if arm == "U0P0":
        cost_u = np.asarray(historical["cost_U"], dtype=np.float64)
        cost_w = np.asarray(historical["cost_W"], dtype=np.float64)
        u1_sha = None
    else:
        u1 = json.loads((Path(u1_dir) / "FIT_U1.json").read_text())
        if u1["anchor"] != anchor or u1["selection_household_assignment_sha256"] != split_sha:
            raise ValueError("frozen U1 decoder selection mismatch")
        decoder_dir = Path(u1_dir) / u1["selected_model_relative"]
        if _model_hash(decoder_dir) != u1["selected_model_sha256"]:
            raise ValueError("frozen U1 decoder hash mismatch")
        from experiments.pcrl_task_directed_release_v1.audits import load_candidate
        decoder = load_candidate(decoder_dir)
        coeff = data.coefficient_pool(prepared, data.load_map(value, anchor, "D17"))
        mechanism = np.asarray(prepared["roles"]["mechanism"], dtype=np.int64)
        t = _codes(prepared, "representation_fit", mechanism)
        mask, _ = _valid_task_rows(coeff)
        pair = method.task_cost_from_decoder(decoder, coeff["ha"][mask], t[mask],
            np.asarray(coeff["labels"]["same_residence"])[mask], coeff["weights"][mask])
        cost_u, cost_w = pair["U"], pair["W"]
        u1_sha = u1["selected_model_sha256"]
    cost = .5*(cost_u+cost_w)
    roles = {name: law for name, law in historical["roles"].items()
             if name.startswith("A/")}
    if set(roles) != {"A/SEX/U", "A/SEX/W", "A/RAC1P/U", "A/RAC1P/W"}:
        raise ValueError("historical local privacy role schema changed")
    root = _private_output(output_dir)
    np.savez_compressed(root / "cost.npz", cost_U=cost_u, cost_W=cost_w, cost=cost)
    if arm == "U0P0" and budget == 0.01:
        q = data.load_map(value, anchor, "Q")
        from experiments.pcrl_task_directed_release_v1 import finite
        cmi = {name: finite.cmi(law, q) for name, law in roles.items()}
        feasible = all(leak <= budget+finite.ACCEPTANCE_TOLERANCES["cmi"]
                       for leak in cmi.values())
        solved = {"Q": q, "objective": float(np.sum(cost*q)),
                  "status": "reused_historical_Q", "feasible": feasible,
                  "residuals": {"cmi": cmi}, "solver": "archived_CLARABEL_or_SCS"}
    else:
        solved = solver.solve_p0(cost, roles, budget,
                                 state_mass=historical["state_mass"],
                                 parent=np.arange(32), zero_action=0)
    channel_relative = None
    if solved.get("Q") is not None and solved.get("feasible"):
        channel_relative = "channel"
        release.ChannelArtifact(solved["Q"],
                                data.member_record(value, anchor, "encoder")["sha256"],
                                f"{arm}_CMI{budget:g}_anchor_{anchor}",
                                "EXPERIMENTAL_UNVALIDATED").save(root / channel_relative)
    record = {"schema": 1, "anchor": anchor, "arm": arm, "budget_nats": budget,
              "policy": "historical A-only empirical conditional information",
              "roles": sorted(roles),
              "status": solved["status"], "feasible": bool(solved.get("feasible")),
              "channel_relative": channel_relative,
              "channel_sha256": data.sha256_file(root / channel_relative / "Q.npz")
              if channel_relative else None,
              "cost_sha256": data.sha256_file(root / "cost.npz"),
              "u1_selected_model_sha256": u1_sha,
              "selection_household_assignment_sha256": split_sha,
              "solution": {k: v for k, v in solved.items() if k != "Q"},
              "created_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(root / "FIT_P0_ARM.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    u1 = sub.add_parser("u1")
    bank = sub.add_parser("bank")
    arm = sub.add_parser("arm")
    p0 = sub.add_parser("p0")
    center = sub.add_parser("center")
    for p in (u1, bank, arm, p0, center):
        p.add_argument("--anchor", type=int, required=True)
        p.add_argument("--index", required=True)
        p.add_argument("--output", required=True)
    center.add_argument("--u1-dir", required=True)
    center.add_argument("--source", choices=SOURCES, action="append")
    bank.add_argument("--source", choices=SOURCES, action="append")
    bank.add_argument("--attack-slate", choices=("standard", "catchup"), default="standard")
    center.add_argument("--attack-slate", choices=("standard", "catchup"), default="standard")
    bank.add_argument("--resume", action="store_true")
    center.add_argument("--resume", action="store_true")
    arm.add_argument("--bank-dir", required=True)
    arm.add_argument("--u1-dir")
    arm.add_argument("--arm", choices=("U0P1", "U1P1"), required=True)
    arm.add_argument("--delta", type=float, required=True)
    p0.add_argument("--arm", choices=("U0P0", "U1P0"), required=True)
    p0.add_argument("--budget", type=float, required=True)
    p0.add_argument("--u1-dir")
    args = parser.parse_args(argv)
    if args.command == "u1":
        record = fit_u1(args.anchor, args.index, args.output)
    elif args.command == "bank":
        record = fit_reference_bank(args.anchor, args.index, args.output,
                                    source_channels=tuple(args.source) if args.source else SOURCES,
                                    attack_slate=args.attack_slate, resume=args.resume)
    elif args.command == "arm":
        record = solve_from_bank(args.anchor, args.index, args.bank_dir, args.output,
                                 arm=args.arm, delta=args.delta, u1_dir=args.u1_dir)
    elif args.command == "p0":
        record = solve_p0_arm(args.anchor, args.index, args.output,
                              arm=args.arm, budget=args.budget, u1_dir=args.u1_dir)
    else:
        record = fit_p1_center(args.anchor, args.index, args.u1_dir, args.output,
                               source_channels=tuple(args.source) if args.source else SOURCES,
                               attack_slate=args.attack_slate, resume=args.resume)
    print(json.dumps({k: v for k, v in record.items() if k not in ("solution", "role_support")},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
