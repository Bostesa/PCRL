"""Fresh complete-wire reader banks, trained on the actual finite release law.

This module has no all-role loader and never accepts INNER labels while
fitting. Numerical entry points require the coordinator's queue guard. The
caller owns admission, exact source/role hashing and global stage locks.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .metrics import integrated_auc, validate_score_law

READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32", "CELL_A1",
           "CODE_BIT_LR_C1", "PROB_LR_C1", "PROB_HGB_15x200")
VIEWS = ("r1", "r2", "pair")
SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"
TRAIN_SAMPLE_DRAWS = 4
TRAIN_SAMPLE_SEED_BASE = 310700
TIE = 1e-12
SHUFFLE_ATT_SALT = "pcrl-method-portfolio-v1|shuffled-sex-probe|20261008"
SHUFFLE_INNER_SALT = "pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2"


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def save_private_npz(path, **arrays):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def attack_split(groups, *, require_adult=True):
    g = np.asarray(groups, dtype=np.int64)
    unique = np.unique(g)
    if require_adult and len(unique) != 4061:
        raise ValueError("ATTACK_FIT_NEW requires 4061 duplicate groups")
    tune_count = 1015 if require_adult else max(2, len(unique) // 4)
    ranked = sorted((hashlib.sha256((SPLIT_SALT + str(int(group))).encode()).hexdigest(), int(group)) for group in unique)
    tune_groups = np.array([group for _, group in ranked[:tune_count]], dtype=np.int64)
    tune = np.isin(g, tune_groups)
    if not tune.any() or tune.all():
        raise ValueError("empty ATT train or tune groups")
    return {"fit": np.flatnonzero(~tune), "tune": np.flatnonzero(tune),
            "fit_groups": len(unique) - tune_count, "tune_groups": tune_count}


def law_arrays(law, bank):
    c1, c2, mass = (np.asarray(law[name]) for name in ("code_r1", "code_r2", "mass"))
    b1, b2 = (np.asarray(bank[f"recipients"][str(r)]["probability"], dtype=np.float64) for r in (1, 2))
    n = len(law["row_id"])
    if (c1.dtype != np.int64 or c2.dtype != np.int64 or mass.dtype != np.float64 or
            c1.ndim != 2 or c1.shape != c2.shape or c1.shape != mass.shape or c1.shape[0] != n or
            np.any(c1 < 0) or np.any(c1 >= len(b1)) or np.any(c2 < 0) or np.any(c2 >= len(b2)) or
            not np.isfinite(mass).all() or np.any(mass < 0) or not np.allclose(mass.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError("actual global-code finite law invalid")
    row, group = np.asarray(law["row_id"]), np.asarray(law["unit"])
    if row.dtype != np.int64 or row.shape != (n,) or group.dtype != np.int64 or group.shape != row.shape or len(np.unique(row)) != n:
        raise ValueError("law private alignment identities invalid")
    return b1, b2, {"r1": c1, "r2": c2, "pair": c1 * len(b2) + c2}, mass


def wire_feature_tables(bank, uniform_public_pair_mass):
    """Lossless public-ID encoding; public full-bank vocabulary is label-free.

    The original source reader used ATT-only categorical vocabularies. Here
    the already public global bank fixes them prospectively for every arm.
    All varying fields are deterministic functions of public global IDs.
    Constant schema/string metadata is preserved literally in the contract;
    it supplies no person-varying signal and is not silently erased.
    """
    b = {r: np.asarray(bank["recipients"][str(r)]["probability"], np.float64) for r in (1, 2)}
    joint = np.asarray(uniform_public_pair_mass, dtype=np.float64)
    if joint.shape != (len(b[1]), len(b[2])) or not np.isfinite(joint).all() or np.any(joint < 0) or not np.isclose(joint.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("public uniform input-prior kernel output law invalid")
    result, schemas, bits = {}, {}, {}
    for recipient in (1, 2):
        probability = b[recipient]
        size, classes = probability.shape
        width = (size - 1).bit_length()
        code = np.arange(size, dtype=np.int64)
        bit = ((code[:, None] >> np.arange(max(1, width))) & 1).astype(float)
        bits[recipient] = bit
        decision = np.eye(classes)[probability.argmax(1)]
        x = np.column_stack((np.eye(size), bit, probability, decision,
                             np.full(size, recipient), np.full(size, width)))
        marginal = joint.sum(axis=1 if recipient == 1 else 0)
        result[f"r{recipient}"] = {"x": x, "bits": bit, "probability": probability,
            "defense": np.column_stack((x, np.log(np.maximum(probability, 1e-12)),
                                         np.log(np.maximum(marginal, 1e-12))))}
        schemas[f"r{recipient}"] = {
            "varying_fields": ["code_id", "probability", "decision"],
            "derived_fields": ["code_bits", "public_log_probability", "public_uniform_source_log_output_mass"],
            "wire_fields": ["schema", "recipient", "probability", "decision", "release_id", "code_id", "public_metadata"],
            "public_metadata_fields": ["model_id", "source_binding_sha256", "bank_sha256", "code_length_bits"],
            "constant_fields": {"schema":"SCV1-reference-score-release-v1", "recipient":recipient,
                "release_id":bank["release_id"], "model_id":bank["release_id"],
                "source_binding_sha256":bank["source_binding_sha256"],
                "bank_sha256":bank["recipients"][str(recipient)].get("bank_sha256"), "code_length_bits":width},
            "code_vocabulary": "all frozen public original global IDs in bank order",
            "feature_order": ["code_onehot", "code_bits", "frozen_probability", "decision_onehot", "recipient", "code_length_bits"],
            "public_side_information": "frozen code, kernel parameters and bank; provider inputs remain private",
            "uniform_prior": "each possible original input code pair once, no label or empirical provider prior"}
    left = np.repeat(np.arange(len(b[1])), len(b[2])); right = np.tile(np.arange(len(b[2])), len(b[1]))
    x = np.column_stack((result["r1"]["x"][left], result["r2"]["x"][right]))
    bit_product = (bits[1][left, :, None] * bits[2][right, None, :]).reshape(len(left), -1)
    result["pair"] = {"x":x, "bits":np.column_stack((bits[1][left], bits[2][right])),
        "probability":np.column_stack((b[1][left], b[2][right])),
        "defense":np.column_stack((result["r1"]["defense"][left], result["r2"]["defense"][right],
                                    bit_product, np.log(np.maximum(joint.ravel(), 1e-12))))}
    schemas["pair"] = {"left":schemas["r1"], "right":schemas["r2"],
        "extra_derived_fields":["pair_code_bit_products", "pair_public_uniform_source_log_output_mass"],
        "no_private_inputs":True, "no_action_positions_or_coins":True}
    return result, schemas


def aggregated_training(index, mass, labels, positions, bank_size):
    count = np.zeros((bank_size, 2), dtype=np.float64)
    idx, prob, y = index[positions], mass[positions], labels[positions]
    np.add.at(count, (idx.ravel(), np.broadcast_to(y[:, None], idx.shape).ravel()), prob.ravel())
    wire, label = np.nonzero(count > 0)
    weight = count[wire, label]
    if np.unique(label).size != 2:
        raise ValueError("reader fitting law lacks both SEX classes")
    return wire, label, weight, count


def sample_training_actions(law, *, seed):
    """Four independent complete-law draws; duplicates share a draw cache."""
    if seed not in (0, 1):
        raise ValueError("paired source seed must be 0 or 1")
    mass = np.asarray(law["mass"])
    groups = np.asarray(law["unit"])
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    for name in ("mass", "code_r1", "code_r2"):
        actual = np.asarray(law[name])
        if not np.array_equal(actual, actual[first][inverse]):
            raise ValueError("duplicate cache law action ordering/masses differ")
    rng = np.random.default_rng(np.random.SeedSequence([TRAIN_SAMPLE_SEED_BASE + seed, 0]))
    cumulative = np.cumsum(mass[first], axis=1); cumulative[:, -1] = 1
    uniform = rng.random((len(unique), TRAIN_SAMPLE_DRAWS))
    slots = np.sum(uniform[:, :, None] >= cumulative[:, None, :], axis=2)
    return slots[inverse]


def fit_weighted(table, wire, label, weight, name, seed):
    if "HGB" in name:
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, max_iter=200,
                                               early_stopping=False, random_state=seed)
        model.fit(table[wire], label, sample_weight=weight)
        return {"model":model, "scaler":None}
    scaler = StandardScaler().fit(table[wire], sample_weight=weight)
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    model.fit(scaler.transform(table[wire]), label, sample_weight=weight)
    return {"model":model, "scaler":scaler}


def predict_model(record, table):
    x = record["scaler"].transform(table) if record["scaler"] is not None else table
    return np.asarray(record["model"].predict_proba(x)[:, 1], np.float64)


def group_shuffle(labels, groups, salt):
    if salt not in (SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT):
        raise ValueError("unregistered control shuffle salt")
    y, g = np.asarray(labels), np.asarray(groups)
    if y.ndim != 1 or g.shape != y.shape:
        raise ValueError("aligned group labels required")
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], "big")))
    buckets = {}
    for group in np.unique(g):
        indexes = np.flatnonzero(g == group)
        buckets.setdefault(len(indexes), []).append(indexes)
    out = np.empty_like(y)
    for size in sorted(buckets):
        values = buckets[size]
        for target, source in zip(values, rng.permutation(len(values))):
            out[target] = y[values[int(source)]]
    return out


def fit_complete_readers(bank, law, sex_record, *, uniform_public_pair_mass, seed, private_root, queue_guard,
                         require_adult=True, source_bindings=None):
    """Freeze all models/orientations without seeing reporting SEX or outcomes."""
    if not callable(queue_guard):
        raise PermissionError("active global queue guard required")
    queue_guard()
    root = Path(private_root)
    if ".private" not in root.resolve().parts or root.exists():
        raise PermissionError("fresh private reader directory required")
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.mkdir(mode=0o700)
    b1, b2, indexes, mass = law_arrays(law, bank)
    for name in ("row_id", "unit"):
        if not np.array_equal(law[name], sex_record[name]):
            raise PermissionError("ATT labels and release people differ")
    y = np.asarray(sex_record["sex"])
    validate_score_law(indexes["pair"].astype(float), mass, y, law["unit"])
    split = attack_split(law["unit"], require_adult=require_adult)
    fit, tune = split["fit"], split["tune"]
    if np.unique(y[fit]).size != 2 or np.unique(y[tune]).size != 2:
        raise ValueError("ATT split lacks both SEX classes")
    tables, schema = wire_feature_tables(bank, uniform_public_pair_mass)
    save_private_npz(root / "PUBLIC_UNIFORM_PRIOR.npz", pair=np.asarray(uniform_public_pair_mass, dtype=np.float64))
    if "sample_weight" not in inspect.signature(MLPClassifier.fit).parameters:
        raise RuntimeError("registered exact weighted MLP requires sklearn sample_weight support")
    save_private_npz(root / "TRAINING_INPUTS.npz", **law, sex=y, fit=fit, tune=tune)
    frozen_tables, validation, model_files, training_receipts = {}, {}, {}, []
    for view in VIEWS:
        validation[view] = {}
        ids = indexes[view]
        tab = tables[view]
        wire, label, weight, counts = aggregated_training(ids, mass, y, fit, len(tab["x"]))
        save_private_npz(root / f"TABLE_INPUTS_{view}.npz", **tab, weighted_counts=counts)
        for number, name in enumerate(READERS):
            queue_guard(); start, cpu = time.perf_counter(), time.process_time()
            kind = "defense" if name.startswith("DA_") else "bits" if name == "CODE_BIT_LR_C1" else "probability" if name.startswith("PROB_") else "x"
            table = tab[kind]
            caught = []
            if name == "CELL_A1":
                prior = float((counts[:, 1].sum() + 1) / (counts.sum() + 2))
                pred = np.full(len(counts), prior)
                seen = counts.sum(1) > 0
                pred[seen] = (counts[seen, 1] + 1) / (counts[seen].sum(1) + 2)
                model = {"counts":counts, "prior":prior, "alpha":1.0}
                training_kind = "exact law weighted categorical counts"
            elif "MLP" in name:
                scaler = StandardScaler().fit(table[wire], sample_weight=weight)
                model = {"scaler":scaler, "model":MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    solver="adam", alpha=1e-4, max_iter=300, early_stopping=False,
                    random_state=TRAIN_SAMPLE_SEED_BASE + seed)}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model["model"].fit(scaler.transform(table[wire]), label, sample_weight=weight)
                pred = predict_model(model, table)
                training_kind = "exact law weighted emitted-wire-by-SEX aggregation; weighted MLP minibatches"
            else:
                train_wire, train_label, train_weight = wire, label, weight
                if "HGB" in name:
                    # HGB's min_samples_leaf counts physical training rows.
                    # Collapsing 14 income bank codes to <=28 wire-label cells
                    # would make the inherited 20-row threshold almost null.
                    # Retain each person/action with its actual positive law
                    # weight; no averaged-probability feature is introduced.
                    positive = mass[fit] > 0
                    train_wire = ids[fit][positive]
                    train_label = np.broadcast_to(y[fit, None], positive.shape)[positive]
                    train_weight = mass[fit][positive]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = fit_weighted(table, train_wire, train_label, train_weight, name, TRAIN_SAMPLE_SEED_BASE + seed)
                pred = predict_model(model, table)
                training_kind = ("exact law weighted positive person-action rows" if "HGB" in name else
                                 "exact law weighted emitted-wire-by-SEX aggregation")
            raw = integrated_auc(pred[ids[tune]], mass[tune], y[tune], groups=law["unit"][tune])
            orientation = 1 if raw >= 0.5 else -1
            frozen_tables[f"{view}__{name}"] = pred if orientation == 1 else 1 - pred
            validation[view][name] = {"positive_auc":raw, "selected_orientation":"positive" if orientation == 1 else "reverse",
                "fixed_orientation_auc":max(raw, 1 - raw), "training":training_kind}
            path = root / f"MODEL_{view}__{name}.joblib"
            joblib.dump(model, path, compress=3); path.chmod(0o600)
            model_files[path.name] = sha_file(path)
            training_receipts.append({"view":view, "reader":name, "table_kind":kind,
                "law_weight_sum":float(weight.sum()), "weighted_training_cells":len(wire),
                "sampled_training_rows":0,
                "wall_seconds":time.perf_counter() - start, "cpu_seconds":time.process_time() - cpu,
                "warnings":[type(item.message).__name__ + ": " + str(item.message) for item in caught]})
        # Real shuffled-label and planted-bit probes are separate engineering
        # copies, and cannot enter the candidate reader slate.
        queue_guard()
        shuffled = group_shuffle(y[fit], law["unit"][fit], SHUFFLE_ATT_SALT)
        w, l, v, _ = aggregated_training(ids[fit], mass[fit], shuffled, np.arange(len(fit)), len(tab["x"]))
        shuffle_model = fit_weighted(tab["x"], w, l, v, "LR_C1", 0)
        frozen_tables[f"{view}__SHUFFLED"] = predict_model(shuffle_model, tab["x"])
        planted_table = np.column_stack((np.repeat(tab["x"], 2, axis=0), np.tile((0., 1.), len(tab["x"]))))
        # Each weighted row receives its true fitting-label engineering bit.
        planted_wire = wire * 2 + label
        planted_model = fit_weighted(planted_table, planted_wire, label, weight, "LR_C1", 0)
        frozen_tables[f"{view}__PLANTED"] = predict_model(planted_model, planted_table).reshape(-1, 2)
        for kind, model in (("SHUFFLED", shuffle_model), ("PLANTED", planted_model)):
            path = root / f"MODEL_{view}__{kind}.joblib"
            joblib.dump(model, path, compress=3); path.chmod(0o600)
            model_files[path.name] = sha_file(path)
    # The aligned pair envelope has every local reader as a zero-fit alias.
    left = np.repeat(np.arange(len(b1)), len(b2)); right = np.tile(np.arange(len(b2)), len(b1))
    for local, codes, ignore in (("r1", left, "R2"), ("r2", right, "R1")):
        for name in READERS:
            frozen_tables[f"pair__IGNORE_{ignore}__{name}"] = frozen_tables[f"{local}__{name}"][codes]
    path = root / "FROZEN_SCORE_TABLES.npz"
    save_private_npz(path, **frozen_tables)
    receipt = {"schema":"pcrl-conditional-fresh-complete-wire-reader-freeze-v1", "status":"FROZEN_COMPLETE",
        "source_seed":seed, "source_binding_sha256":bank["source_binding_sha256"],
        "source_bindings":source_bindings, "reader_order":list(READERS),
        "score_table_sha256":sha_file(path), "input_sha256":sha_file(root / "TRAINING_INPUTS.npz"),
        "public_uniform_prior_sha256":sha_file(root / "PUBLIC_UNIFORM_PRIOR.npz"),
        "fit_groups":split["fit_groups"], "tune_groups":split["tune_groups"], "fit_rows":len(fit), "tune_rows":len(tune),
        "reader_model_seed":TRAIN_SAMPLE_SEED_BASE + seed, "reader_training_draws":0,
        "weighted_MLP_batch_note":"minibatches contain aggregated emitted-wire-by-SEX cells; actual sample weights enter fit; no sampled or mean outputs",
        "orientation":"ATT_TUNE only, frozen before INNER SEX; ties choose positive",
        "primary_envelope":"max frozen-oriented declared reader AUC per seed/view; no INNER orientation selection",
        "model_files":model_files, "numerical_fit_count":30, "ignore_other_new_fits":0,
        "complete_wire_contract":schema, "validation":validation, "training_receipts":training_receipts,
        "weighted_HGB_sample_count_note":"min_samples_leaf=20 counts positive person-action rows; sample_weight carries exact person-law mass; identical rule across arms",
        "sklearn_version":__import__("sklearn").__version__, "numpy_version":np.__version__}
    queue_guard()
    write_private_json(root / "FROZEN_COMPLETE.json", receipt)
    return receipt


def score_frozen_readers(bank, law, sex_record, frozen_tables):
    """Reporting only: caller must establish the global all-reader freeze first."""
    _, _, indexes, mass = law_arrays(law, bank)
    for name in ("row_id", "unit"):
        if not np.array_equal(law[name], sex_record[name]):
            raise PermissionError("reporting people/labels differ")
    y = np.asarray(sex_record["sex"])
    validate_score_law(indexes["pair"].astype(float), mass, y, law["unit"])
    shuffled = group_shuffle(y, law["unit"], SHUFFLE_INNER_SALT)
    result, laws = {}, {}
    for view in VIEWS:
        prefix = view + "__"
        scores = {name[len(prefix):]:np.asarray(table)[indexes[view]] for name, table in frozen_tables.items()
                  if name.startswith(prefix) and name[len(prefix):] not in ("SHUFFLED", "PLANTED")}
        reader_auc = {name:integrated_auc(value, mass, y, groups=law["unit"]) for name, value in scores.items()}
        null = integrated_auc(np.full_like(mass, 0.5), mass, y, groups=law["unit"])
        shuffle_score = frozen_tables[prefix + "SHUFFLED"][indexes[view]]
        shuffle_auc = integrated_auc(shuffle_score, mass, shuffled, groups=law["unit"])
        # The planted true-label bit is used ONLY on a separate engineering copy.
        planted = frozen_tables[prefix + "PLANTED"][indexes[view], y[:, None]]
        planted_auc = integrated_auc(planted, mass, y, groups=law["unit"])
        controls = {"null_auc":null, "shuffled_independent_inner_auc":max(shuffle_auc, 1 - shuffle_auc),
                    "planted_auc":planted_auc}
        gates = {"null_exact":null == 0.5, "shuffled_at_most_0.60":controls["shuffled_independent_inner_auc"] <= 0.60,
                 "planted_at_least_0.99":planted_auc >= 0.99}
        result[view] = {"reader_auc":reader_auc, "envelope_auc":max(reader_auc.values()),
                        "controls":controls, "control_gates":gates, "controls_pass":all(gates.values())}
        laws[view] = scores
    monotonic = result["pair"]["envelope_auc"] + 1e-12 >= max(result["r1"]["envelope_auc"], result["r2"]["envelope_auc"])
    if not monotonic:
        raise AssertionError("ignore-other coalition envelope is not monotone")
    return {"views":result, "ignore_other_monotonicity":monotonic,
            "all_controls_pass":all(v["controls_pass"] for v in result.values())}, laws
