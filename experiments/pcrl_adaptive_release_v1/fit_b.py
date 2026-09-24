"""Branch B nested-state fitting primitives on frozen 2018 development roles.

Scientific dispatch is owned by the coordinator. This module is import-safe:
it never loads archived ACS rows or starts a fit at import time. The stored
historical T0 codes are used for 2018 fitting; Mac float32 re-encoding cannot
replace their Linux x86 bytes. Deployable routing must use a verified
compatible encoder/runtime, as documented by the new study protocol.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from . import refinement

MIN_CHECK_GAIN_RATIO = 0.25  # frozen before any Branch B ACS candidate fit
AMENDMENT_01_SHA256 = "a77279820a0996159b99ea56c9a97eabf51179a0c5c722f5a9283ed2f569e0a8"


def _source_provenance() -> dict:
    """Pin the registered amendment, checkout and changing B source bytes."""
    from experiments.pcrl_task_aligned_cuts_v1 import data

    workspace = Path(__file__).resolve().parents[2]
    amendment = workspace / "results/pcrl_adaptive_release_v1/AMENDMENT_01.md"
    if not amendment.is_file() or data.sha256_file(amendment) != AMENDMENT_01_SHA256:
        raise ValueError("registered pre-B amendment is absent or changed")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace,
                            check=True, capture_output=True, text=True).stdout.strip()
    if len(commit) != 40:
        raise ValueError("full scientific source commit required")
    module = Path(__file__).resolve().parent
    return {"amendment_01_sha256": AMENDMENT_01_SHA256,
            "source_commit": commit,
            "fit_b_source_sha256": data.sha256_file(module / "fit_b.py"),
            "refinement_source_sha256": data.sha256_file(module / "refinement.py"),
            "nuisance_source_sha256": data.sha256_file(module / "nuisance.py")}


def _array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(a.shape).encode())
    digest.update(str(a.dtype).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def runtime_parity_receipt(stored_rows: Mapping, partition: refinement.NestedPartition,
                           frozen_encoder, frozen_nuisance, *,
                           encoder_sha256: str, nuisance_sha256: str,
                           q: np.ndarray | None = None) -> dict:
    """Compare archived Linux T0/child IDs with live local-input routing.

    Run on compatible Linux x86 before production. A Mac diagnostic records
    differences but cannot set `linux_x86_parity_verified` even if the small
    fixture happens to agree. No protected/task label is read here.
    """
    if len(encoder_sha256) != 64 or len(nuisance_sha256) != 64:
        raise ValueError("pinned frozen encoder/nuisance hashes required")
    archived_t, archived_features = stored_deployable_features(stored_rows, frozen_nuisance)
    inputs = RuntimeInputs(np.asarray(stored_rows["x"]), np.asarray(stored_rows["ha"]))
    runtime_t, runtime_features = refinement.build_deployable_features(
        inputs, frozen_encoder, frozen_nuisance)
    code_equal = bool(np.array_equal(runtime_t, archived_t))
    runtime_child = partition.route(runtime_t, runtime_features)
    archived_child = partition.route(archived_t, archived_features)
    child_equal = bool(np.array_equal(runtime_child, archived_child))
    service_equal = bool(np.asarray(inputs.h_a).tobytes() ==
                         np.asarray(stored_rows["ha"]).tobytes())
    p_diff = float(np.max(np.abs(runtime_features["task_posterior"] -
                                 archived_features["task_posterior"])))
    r_diff = float(np.max(np.abs(runtime_features["residual"] -
                                 archived_features["residual"])))
    linux = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
    return {"schema": "pcrl-refined-runtime-parity-v1",
            "system": platform.system(), "machine": platform.machine(),
            "encoder_sha256": encoder_sha256,
            "nuisance_sha256": nuisance_sha256,
            "channel_array_sha256": None if q is None else _array_sha256(q),
            "partition_record": partition.to_record(),
            "fixture_input_sha256": _array_sha256(inputs.x_a),
            "fixture_h_a_sha256": _array_sha256(inputs.h_a),
            "checked_original_people": int(len(archived_t)),
            "t32_bitwise_equal": code_equal,
            "child_bitwise_equal": child_equal,
            "service_byte_equal": service_equal,
            "task_posterior_max_absolute_difference": p_diff,
            "residual_max_absolute_difference": r_diff,
            "linux_x86_parity_verified": bool(linux and code_equal and child_equal
                                               and service_equal and p_diff <= 1e-7
                                               and r_diff <= 1e-7 and q is not None)}


class RefinedReleaseSession:
    """One-token local release with an in-process immutable-input cache.

    The stored historical encoder and new frozen nuisance are both private
    runtime dependencies. The public wire is only unchanged H_A plus Z.
    Actual deployment requires a Linux x86 parity receipt for T0 and child
    routing. `synthetic_fixture` is solely for public synthetic smoke tests.
    """

    def __init__(self, q: np.ndarray, partition: refinement.NestedPartition,
                 frozen_encoder, frozen_nuisance, *,
                 parity_receipt: Mapping | None = None,
                 encoder_sha256: str | None = None,
                 nuisance_sha256: str | None = None,
                 synthetic_fixture: bool = False,
                 replay_key: bytes | None = None,
                 rng: np.random.Generator | None = None):
        q = np.asarray(q, dtype=np.float64)
        if (q.shape != (len(partition.parents), 17) or
                not np.isfinite(q).all() or np.any(q < 0) or
                np.max(np.abs(q.sum(axis=1)-1)) > 1e-10):
            raise ValueError("refined channel must have normalized child×17 rows")
        if not synthetic_fixture:
            if (platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64") or
                    parity_receipt is None or
                    parity_receipt.get("linux_x86_parity_verified") is not True or
                    parity_receipt.get("partition_record") != partition.to_record() or
                    parity_receipt.get("encoder_sha256") != encoder_sha256 or
                    parity_receipt.get("nuisance_sha256") != nuisance_sha256 or
                    parity_receipt.get("channel_array_sha256") != _array_sha256(q) or
                    parity_receipt.get("t32_bitwise_equal") is not True or
                    parity_receipt.get("child_bitwise_equal") is not True or
                    parity_receipt.get("service_byte_equal") is not True):
                raise ValueError("verified Linux x86 parity required for refined release")
            if replay_key is None:
                raise ValueError("private replay key required for restart-stable one-release token")
        if replay_key is not None and (not isinstance(replay_key, bytes) or len(replay_key) < 32):
            raise ValueError("private replay key must contain at least 32 bytes")
        self._q = q.copy()
        self._q.flags.writeable = False
        self._partition = partition
        self._encoder = frozen_encoder
        self._nuisance = frozen_nuisance
        self._rng = np.random.default_rng() if rng is None else rng
        self._replay_key = replay_key
        self._channel_identity = _array_sha256(q) + hashlib.sha256(
            json.dumps(partition.to_record(), sort_keys=True).encode()).hexdigest()
        self._cache: dict[object, tuple[str, int]] = {}

    @staticmethod
    def _input_digest(x: np.ndarray, h: np.ndarray) -> str:
        digest = hashlib.sha256()
        for value in (x, h):
            a = np.ascontiguousarray(value)
            digest.update(str(a.dtype).encode())
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def private_expected_law(self, inputs: RuntimeInputs) -> np.ndarray:
        """For exact expected-token audit only; never serialize to the wire."""
        leaf = self._partition.route_runtime(inputs, self._encoder, self._nuisance)
        return self._q[leaf].copy()

    def release(self, inputs: RuntimeInputs, *, person_ids: Sequence) -> dict:
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("release requires only RuntimeInputs(X_A,H_A)")
        ids = list(person_ids)
        if len(ids) != len(inputs.h_a) or len({str(i) for i in ids}) != len(ids):
            raise ValueError("one distinct private cache ID per original person required")
        law = self.private_expected_law(inputs)
        token = np.empty(len(ids), dtype=np.int64)
        for i, identifier in enumerate(ids):
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or identifier == "":
                raise ValueError("private cache ID must be a nonempty string or integer")
            digest = self._input_digest(np.asarray(inputs.x_a)[i],
                                        np.asarray(inputs.h_a)[i])
            if identifier in self._cache:
                prior_digest, prior_token = self._cache[identifier]
                if digest != prior_digest:
                    raise ValueError("immutable input snapshot changed for cached person")
                token[i] = prior_token
            else:
                if self._replay_key is None:
                    chosen = int(self._rng.choice(17, p=law[i]))
                else:
                    payload = json.dumps({"id_type": type(identifier).__name__,
                                          "id": identifier, "input": digest,
                                          "channel": self._channel_identity},
                                         sort_keys=True, separators=(",", ":")).encode()
                    draw = hmac.new(self._replay_key, payload, hashlib.sha256).digest()
                    u = (int.from_bytes(draw, "big") + .5)/2**256
                    chosen = min(16, int(np.searchsorted(np.cumsum(law[i]), u, side="right")))
                self._cache[identifier] = (digest, chosen)
                token[i] = chosen
        return {"h_a": np.asarray(inputs.h_a).copy(), "token": token}


def load_a_selected(a_center_dir: str | Path, *, anchor: int, delta: float) -> dict:
    """Read complete Branch A evidence; reuse rather than refit its T32 slates."""
    from experiments.pcrl_task_aligned_cuts_v1 import data, release, solver
    from experiments.pcrl_task_aligned_cuts_v1.audit import model_directory_hash
    from experiments.pcrl_task_directed_release_v1.audits import load_candidate
    from . import fit_a

    root = Path(a_center_dir)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if (complete.get("schema") != "pcrl-adaptive-A-center-v1" or
            complete.get("status") != "COMPLETE" or
            complete.get("anchor") != anchor or complete.get("delta") != delta or
            complete.get("outer_labels_accessed") is not False):
        raise ValueError("incompatible complete Branch A center required")
    if complete.get("artifact_sha256") != fit_a._inventory(root):
        raise ValueError("Branch A center artifact inventory hash mismatch")
    selected = json.loads((root / "SELECTED.json").read_text())
    round_index = selected["selected_round"]
    if round_index != complete["selected_round"]:
        raise ValueError("Branch A selection and completion disagree")
    final_index = len(complete["rounds"])-1
    final_root = root / f"round_r{final_index:02d}"
    final_record = json.loads((final_root / "ROUND.json").read_text())
    if final_record["bank_sha256"] != complete["rounds"][final_index]["bank_sha256"]:
        raise ValueError("Branch A final retained bank changed")
    selection_path = root / "FINAL_BANK_SELECTION.json"
    final_selection = json.loads(selection_path.read_text())
    if (data.sha256_file(selection_path) != selected["final_bank_selection_sha256"] or
            final_selection["selected_round"] != round_index or
            final_selection["final_bank_sha256"] != selected["final_bank_sha256"] or
            final_selection["final_bank_sha256"] != final_record["bank_sha256"]):
        raise ValueError("Branch A selected channel lacks final-bank selection replay")
    selected_root = root / f"round_r{round_index:02d}"
    selected_record = json.loads((selected_root / "ROUND.json").read_text())
    channel = release.ChannelArtifact.load(selected_root / "channel").Q
    if data.sha256_file(selected_root / "channel" / "Q.npz") != selected_record["channel_file_sha256"]:
        raise ValueError("Branch A selected channel SHA differs")
    costs_path = final_root / "COSTS.npz"
    if data.sha256_file(costs_path) != final_record["task_cost_archive_sha256"]:
        raise ValueError("Branch A final fixed cost SHA differs")
    with np.load(costs_path, allow_pickle=False) as archive:
        cost = {"U": archive["cost_U"], "W": archive["cost_W"],
                "balanced": archive["cost"]}
    bank_record = json.loads((final_root / "CALIBRATED_BANK.json").read_text())
    coeff_path = final_root / "CALIBRATED_COEFFICIENTS.npz"
    if (data.sha256_file(coeff_path) != bank_record["coefficients_sha256"] or
            data.sha256_file(final_root / "CALIBRATED_BANK.json") !=
            final_record["calibrated_bank_manifest_sha256"]):
        raise ValueError("Branch A final attack coefficients differ")
    with np.load(coeff_path, allow_pickle=False) as archive:
        cuts = [{**metadata, "coeff": archive[f"cut_{i:04d}"]}
                for i, metadata in enumerate(bank_record["cuts"])]
    replay = solver.replay_p1(channel, cost["balanced"], cuts)
    declared = final_selection["final_bank_checks"][round_index]
    if (not declared["feasible"] or
            replay["bank_sha256"] != final_selection["final_bank_sha256"] or
            abs(replay["maximum_cut_violation"] - declared["maximum_cut_violation"]) > 1e-10 or
            replay["maximum_cut_violation"] > solver.PRIMAL_TOL):
        raise ValueError("Branch A selected Q fails final rebased union bank")
    decoder_round = int(final_record["selected_decoder_id"].split("/")[0].split("_")[1])
    decoder_root = root / f"decoder_r{decoder_round:02d}"
    decoder_record = json.loads((decoder_root / "FIT_DECODER.json").read_text())
    decoder_path = decoder_root / decoder_record["selected_model_relative_directory"]
    if model_directory_hash(decoder_path) != final_record["selected_decoder_model_sha256"]:
        raise ValueError("Branch A final fixed decoder SHA differs")
    decoder = load_candidate(decoder_path)
    specs = {}
    for i in range(final_index + 1):
        bank_dir = root / f"bank_r{i:02d}"
        manifest = json.loads((bank_dir / "ATTACK_BANK.json").read_text())
        if data.sha256_file(bank_dir / "coefficients.npz") != manifest["coefficient_archive_sha256"]:
            raise ValueError("Branch A retained attack bank archive differs")
        for source in manifest["attack_specs"]:
            path = (bank_dir / source["model_relative_directory"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Branch A attack escaped its private unit")
            specs[source["id"]] = {**source, "model_directory": str(path)}
    if any(cut["id"].rsplit("/", 1)[0] not in specs for cut in cuts):
        raise ValueError("Branch A selected cut lacks retained attack model")
    return {"Q": channel, "cost": cost, "cuts": cuts,
            "attack_specs": specs, "decoder": decoder,
            "decoder_model_sha256": final_record["selected_decoder_model_sha256"],
            "selected_round": round_index, "final_bank_round": final_index,
            "selected_bank_sha256": final_record["bank_sha256"],
            "role_input_sha256": complete["role_input_sha256"],
            "encoder_sha256": complete["encoder_sha256"],
            "q_ref_sha256": complete["q_ref_sha256"],
            "historical_q_sha256": complete["historical_q_sha256"],
            "complete_receipt_sha256": data.sha256_file(root / "COMPLETE.json"),
            "outer_labels_accessed": False}


def fixed_bank_dual_prices(cost: np.ndarray, cuts: Sequence[Mapping], *,
                           time_limit_seconds: float | None = None) -> dict:
    """Verify one fixed LP and extract nonnegative privacy Lagrange prices.

    For L_j(Q)>=floor_j, the Lagrangian is C·Q + Σ λ_j(floor_j-L_j(Q));
    therefore per-person split prices use `task - Σ λ*attack`, not plus.
    The certificate pertains only to this frozen finite bank/decoder/rows.
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 17 or not np.isfinite(c).all():
        raise ValueError("finite state×17 task cost required")
    n_states, n_tokens = c.shape
    ids = []
    coeff = []
    floors = []
    for cut in cuts:
        cid = cut.get("id")
        a = np.asarray(cut.get("coeff"), dtype=np.float64)
        floor = float(cut.get("floor"))
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("unique frozen cut IDs required")
        if a.shape != c.shape or not np.isfinite(a).all() or not np.isfinite(floor):
            raise ValueError("frozen cut shape/floor changed")
        ids.append(cid)
        coeff.append(a)
        floors.append(floor)
    a_eq = sparse.kron(sparse.eye(n_states, format="csr"),
                       np.ones((1, n_tokens)), format="csr")
    a_ub = np.stack([-a.ravel() for a in coeff]) if coeff else None
    b_ub = -np.asarray(floors) if coeff else None
    options = {"time_limit": float(time_limit_seconds)} if time_limit_seconds is not None else None
    solved = linprog(c.ravel(), A_ub=a_ub, b_ub=b_ub,
                     A_eq=a_eq, b_eq=np.ones(n_states),
                     bounds=(0, None), method="highs", options=options)
    if not solved.success or solved.x is None:
        return {"status": "unresolved", "solver_message": solved.message,
                "Q": None, "multipliers": None}
    q = np.asarray(solved.x, dtype=float).reshape(c.shape)
    multipliers = np.maximum(-np.asarray(solved.ineqlin.marginals), 0.) if coeff else np.zeros(0)
    reduced = c.copy()
    for lam, a in zip(multipliers, coeff):
        reduced -= lam*a
    lower = float(np.min(reduced, axis=1).sum() + np.dot(multipliers, floors))
    objective = float(np.sum(c*q))
    cut_residual = max((floor-float(np.sum(a*q)) for floor, a in zip(floors, coeff)), default=0.)
    simplex_residual = float(np.max(np.abs(q.sum(1)-1)))
    if q.min() < -1e-9 or cut_residual > 1e-7 or simplex_residual > 1e-9:
        return {"status": "primal_replay_invalid", "solver_message": solved.message,
                "Q": q, "multipliers": dict(zip(ids, multipliers.tolist())),
                "objective": objective, "dual_lower_bound": lower,
                "cut_violation": cut_residual, "simplex_residual": simplex_residual}
    return {"status": "optimal", "solver_message": solved.message,
            "Q": q, "multipliers": dict(zip(ids, multipliers.tolist())),
            "objective": objective, "dual_lower_bound": lower,
            "numerical_gap": objective-lower,
            "cut_violation": max(0., cut_residual),
            "simplex_residual": simplex_residual,
            "not_interval_arithmetic": True,
            "scope": "fixed frozen task decoder and attack bank; oracle/generalization error unresolved"}


def priced_original_person_rows(rows: Mapping, task_mask: np.ndarray,
                                task_losses: np.ndarray,
                                attack_losses: Sequence[Mapping],
                                multipliers: Mapping[str, float]) -> np.ndarray:
    """Build fixed-price g_i(z) without duplicating or double-weighting people.

    Each attack item has `id`, `weighting` U/W, `valid_mask` on all coefficient
    people, and observed-class `losses` for its valid people. Its coefficient
    weights normalize only that role's valid original people. Task cost is
    0.5 U + 0.5 W on task-valid original people.
    """
    w = np.asarray(rows["weights"], dtype=np.float64)
    task_mask = np.asarray(task_mask, dtype=bool)
    task_losses = np.asarray(task_losses, dtype=np.float64)
    if (w.ndim != 1 or task_mask.shape != w.shape or
            task_losses.shape != (int(task_mask.sum()), 17) or
            not np.isfinite(w).all() or np.any(w < 0) or
            not np.isfinite(task_losses).all() or np.any(task_losses < 0) or
            not task_mask.any() or w[task_mask].sum() <= 0):
        raise ValueError("invalid task-valid original-person losses/weights")
    g = np.zeros((len(w), 17), dtype=np.float64)
    task_weights = 0.5/len(task_losses) + 0.5*w[task_mask]/w[task_mask].sum()
    g[task_mask] = task_weights[:, None]*task_losses
    seen = set()
    for attack in attack_losses:
        cid = attack["id"]
        if cid in seen or cid not in multipliers:
            raise ValueError("missing or duplicate fixed-bank dual price")
        seen.add(cid)
        lam = float(multipliers[cid])
        mask = np.asarray(attack["valid_mask"], dtype=bool)
        losses = np.asarray(attack["losses"], dtype=np.float64)
        if (not np.isfinite(lam) or lam < 0 or mask.shape != w.shape or
                losses.shape != (int(mask.sum()), 17) or
                not np.isfinite(losses).all() or np.any(losses < 0) or
                not mask.any() or w[mask].sum() <= 0):
            raise ValueError("invalid attack-valid original-person losses/dual price")
        if attack["weighting"] == "U":
            attack_weights = np.full(len(losses), 1/len(losses))
        elif attack["weighting"] == "W":
            attack_weights = w[mask]/w[mask].sum()
        else:
            raise ValueError("unknown attack weighting")
        g[mask] -= lam*attack_weights[:, None]*losses
    if seen != set(multipliers):
        raise ValueError("priced attack list differs from frozen bank")
    return g


def aggregate_child_problem(rows: Mapping, child_ids: np.ndarray, n_states: int,
                            task_mask: np.ndarray, task_losses: np.ndarray,
                            attacks: Sequence[Mapping]) -> dict:
    """Regroup frozen per-person token losses under private child IDs.

    U and PWGTP normalize separately on each target's valid original people;
    state mass is included here once. `attacks` are frozen model replays, not
    predictions trained on the coefficient rows. No person-level array is
    suitable for the public result manifest.
    """
    from .fit_a import aggregate_loss_coefficients

    child = np.asarray(child_ids)
    w = np.asarray(rows["weights"], dtype=np.float64)
    task_mask = np.asarray(task_mask, dtype=bool)
    task_losses = np.asarray(task_losses, dtype=np.float64)
    if (child.ndim != 1 or child.dtype.kind not in "iu" or
            w.shape != child.shape or task_mask.shape != child.shape or
            task_losses.shape != (int(task_mask.sum()), 17) or
            np.any(child < 0) or np.any(child >= n_states)):
        raise ValueError("invalid child task coefficient alignment")
    pair = aggregate_loss_coefficients(child[task_mask], task_losses,
                                       w[task_mask], n_states)
    bank = []
    ids = set()
    for attack in attacks:
        cid = attack["id"]
        role = attack["role"]
        target = attack["target"]
        mask = np.asarray(attack["valid_mask"], dtype=bool)
        losses = np.asarray(attack["losses"], dtype=float)
        if (not isinstance(cid, str) or not cid or cid in ids or
                role not in ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P") or
                target != role.split("/")[1] or mask.shape != child.shape or
                losses.shape != (int(mask.sum()), 17) or
                attack["class_order"] != list(range(2 if target == "SEX" else 9)) or
                not attack.get("coefficient_pool_sha256")):
            raise ValueError("invalid frozen attack provenance or token losses")
        ids.add(cid)
        attack_pair = aggregate_loss_coefficients(child[mask], losses,
                                                  w[mask], n_states)
        for weighting in ("U", "W"):
            bank.append({"id": f"{cid}/{weighting}", "role": role,
                         "weighting": weighting, "coeff": attack_pair[weighting],
                         "coefficient_pool_sha256": attack["coefficient_pool_sha256"],
                         "class_order": attack["class_order"],
                         "weight_normalization": ("1/n" if weighting == "U"
                                                  else "PWGTP/sum(PWGTP)"),
                         "source_attack_id": cid,
                         "model_sha256": attack.get("model_sha256")})
    return {"task_cost_pair": pair, "bank": bank,
            "person_count": int(len(child)), "n_states": int(n_states),
            "normalization": "per-role valid original people; no second state mass"}


def stored_deployable_features(rows: Mapping, frozen_nuisance) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Use archived T32/r/p and frozen local nuisance on a named development role.

    The caller must obtain `rows` through the hash-verified pooled-role loader;
    labels in that private dictionary are not read by this function.
    """
    required = {"x", "ha", "token_codes", "teacher_p", "residual"}
    if not required.issubset(rows):
        raise ValueError("missing archived local feature or T32 field")
    local = RuntimeInputs(np.asarray(rows["x"]), np.asarray(rows["ha"]))
    t = np.asarray(rows["token_codes"])
    if t.shape != (len(local.x_a),) or t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t >= 32):
        raise ValueError("invalid archived T32 codes")
    features = {"residual": np.asarray(rows["residual"], dtype=float),
                "task_posterior": np.asarray(rows["teacher_p"], dtype=float)}
    features.update({f"h_a_{j}": np.asarray(local.h_a)[:, j]
                     for j in range(4)})
    predicted = frozen_nuisance.probabilities(local)
    for target, prefix, classes in (("SEX", "sex_risk", 2),
                                    ("RAC1P", "race_risk", 9)):
        p = np.asarray(predicted[target], dtype=float)
        if (p.shape != (len(t), classes) or not np.isfinite(p).all() or
                np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-8)):
            raise ValueError("frozen nuisance probability/class schema changed")
        features.update({f"{prefix}_{j}": p[:, j] for j in range(classes)})
    if (set(features) != refinement.ALLOWED_FEATURES or
            any(value.shape != t.shape or not np.isfinite(value).all()
                for value in features.values())):
        raise ValueError("invalid stored deployable feature alignment")
    return t.astype(np.int64, copy=True), features


def select_nested_partition(coefficient_rows: Mapping, checking_rows: Mapping,
                            frozen_nuisance, priced_coefficient_rows: np.ndarray,
                            priced_checking_rows: np.ndarray, *,
                            max_states: int = 64,
                            min_households: int = 100,
                            min_effective_households: float = 100,
                            feature_names: Sequence[str] = tuple(sorted(refinement.ALLOWED_FEATURES)),
                            quantiles: Sequence[float] = (.25, .5, .75),
                            max_candidates_per_feature: int = 3) -> dict:
    """Greedy bounded fixed-price split search with separate checking scores.

    The caller freezes decoder/bank/dual prices before invoking this search.
    Checking scores are recorded; final channel/partition selection belongs on
    the registered inner-selection resource. A positive split score alone is
    never reported as a feasible or independently audited improvement.
    """
    if not isinstance(max_states, int) or not 32 <= max_states <= 128:
        raise ValueError("registered refined state limit is 32..128")
    coefficient_t, coefficient_features = stored_deployable_features(
        coefficient_rows, frozen_nuisance)
    checking_t, checking_features = stored_deployable_features(
        checking_rows, frozen_nuisance)
    g = np.asarray(priced_coefficient_rows, dtype=np.float64)
    check_g = np.asarray(priced_checking_rows, dtype=np.float64)
    if (g.shape != (len(coefficient_t), 17) or
            check_g.shape != (len(checking_t), 17) or
            not np.isfinite(g).all() or not np.isfinite(check_g).all()):
        raise ValueError("fixed-price rows must be original-person × 17 token")
    coefficient_hh = np.asarray(coefficient_rows["households"])
    checking_hh = np.asarray(checking_rows["households"])
    if np.intersect1d(coefficient_hh, checking_hh).size:
        raise ValueError("split-fitting and checking households overlap")
    partition = refinement.NestedPartition.base(32)
    history = []
    rejected = []
    candidate_history = []
    terminal_status = None
    while len(partition.parents) < max_states:
        fit_leaves = partition.route(coefficient_t, coefficient_features)
        check_leaves = partition.route(checking_t, checking_features)
        candidates = []
        for leaf in range(len(partition.parents)):
            candidates.extend(refinement.rank_splits(
                leaf_ids=fit_leaves, leaf=leaf,
                features=coefficient_features, priced_rows=g,
                households=coefficient_hh,
                weights=np.asarray(coefficient_rows["weights"]),
                min_households=min_households,
                min_effective_weight=min_effective_households,
                feature_names=feature_names, quantiles=quantiles,
                max_candidates_per_feature=max_candidates_per_feature,
                checking=(check_leaves, checking_features, check_g, checking_hh)))
        candidates = [candidate for candidate in candidates if candidate.gain > 1e-12]
        ranked = sorted(candidates, key=lambda c: (-c.gain, c.leaf,
                                                  c.feature_name, c.threshold))
        candidate_history.append([{"leaf": c.leaf, "feature_name": c.feature_name,
                                   "threshold": c.threshold,
                                   "fitting_gain": c.gain,
                                   "checking_gain": c.checking_gain,
                                   "children_unique_households": list(c.children_households),
                                   "children_effective_households": list(c.children_effective_weight)}
                                  for c in ranked])
        if not ranked:
            terminal_status = "support_limited" if min_households > 1 else "no_positive_fixed_price_gain"
            break
        eligible = []
        for candidate in ranked:
            if (candidate.checking_gain is None or
                    candidate.checking_gain + 1e-12 <
                    MIN_CHECK_GAIN_RATIO*candidate.gain):
                rejected.append({"leaf": candidate.leaf,
                                 "feature_name": candidate.feature_name,
                                 "threshold": candidate.threshold,
                                 "fitting_gain": candidate.gain,
                                 "checking_gain": candidate.checking_gain,
                                 "reason": "checking_gain_below_fixed_ratio"})
            else:
                eligible.append(candidate)
        if not eligible:
            terminal_status = "checking_unstable"
            break
        best = eligible[0]
        partition = partition.split(best.leaf, best.feature_name, best.threshold)
        history.append({"leaf": best.leaf, "new_leaf": len(partition.parents)-1,
                        "parent_T32": partition.parents[best.leaf],
                        "feature_name": best.feature_name,
                        "training_only_threshold": best.threshold,
                        "fixed_price_fitting_gain": best.gain,
                        "checking_gain": best.checking_gain,
                        "children_unique_households": list(best.children_households),
                        "children_effective_households": list(best.children_effective_weight),
                        "gain_is_primal_feasible_certificate": False})
    fit_leaves = partition.route(coefficient_t, coefficient_features)
    check_leaves = partition.route(checking_t, checking_features)
    status = ("max_states_reached" if len(partition.parents) == max_states else
              terminal_status or "no_positive_fixed_price_gain")
    return {"partition": partition, "coefficient_leaves": fit_leaves,
            "checking_leaves": check_leaves, "split_history": history,
            "rejected_candidates": rejected,
            "candidate_history": candidate_history,
            "status": status, "max_states": max_states,
            "minimum_checking_to_fitting_gain_ratio": MIN_CHECK_GAIN_RATIO,
            "min_unique_households_per_child": min_households,
            "min_effective_households_per_child": min_effective_households,
            "split_score_scope": "fixed decoder/attack dual prices on 2018 development roles; not feasible or held-out audit improvement"}


def _attack_loss_rows(cuts: Sequence[Mapping], specs: Mapping[str, Mapping],
                      rows: Mapping) -> tuple[list[dict], list[dict]]:
    """Replay each frozen attack once; expand only its U/W price records."""
    from . import fit_a

    cached = {}
    priced = []
    unique = []
    for cut in cuts:
        attack_id = cut["id"].rsplit("/", 1)[0]
        if attack_id not in specs:
            raise ValueError("cut lacks retained frozen attack")
        if attack_id not in cached:
            spec = specs[attack_id]
            valid, losses = fit_a.person_loss_from_attack(spec, rows)
            cached[attack_id] = (valid, losses)
            unique.append({"id": attack_id, "role": spec["role"],
                           "target": spec["target"],
                           "valid_mask": valid, "losses": losses,
                           "class_order": spec["class_order"],
                           "coefficient_pool_sha256": cut["coefficient_pool_sha256"],
                           "model_sha256": spec["model_sha256"]})
        valid, losses = cached[attack_id]
        priced.append({"id": cut["id"], "weighting": cut["weighting"],
                       "valid_mask": valid, "losses": losses})
    return priced, unique


def _child_roles(role_dict: Mapping, partition: refinement.NestedPartition,
                 frozen_nuisance) -> dict:
    """Replace private T32 IDs only on fitting/selection roles, never inner check."""
    out = {}
    for name, rows in role_dict.items():
        if name == "inner_check":
            continue
        parent, features = stored_deployable_features(rows, frozen_nuisance)
        child = partition.route(parent, features)
        if not np.array_equal(partition.parent_of_leaf[child], parent):
            raise AssertionError("child routing failed exact historical parent witness")
        out[name] = {**rows, "token_codes": child}
    return out


def _score_inner(decoder, rows: Mapping, q: np.ndarray) -> dict:
    from .fit_a import task_scores
    return task_scores(decoder, rows, q)


def select_b_final_bank(round_channels: Sequence[np.ndarray],
                        round_records: Sequence[Mapping], final_cost: np.ndarray,
                        final_cuts: Sequence[Mapping]) -> dict:
    """Select a channel only after replay on the last rebased union bank."""
    from .fit_a import select_final_bank_feasible
    result = select_final_bank_feasible(round_channels, round_records,
                                        final_cost, final_cuts)
    return {**result,
            "selected_channel": np.asarray(round_channels[result["selected_round"]]).copy(),
            "selection_scope": "inner-selection fitted-decoder surrogate; independent common audit remains separate"}


def run_center_from_roles(anchor: int, delta: float, role_dict: Mapping,
                          q_ref: np.ndarray, historical_q: np.ndarray,
                          encoder_sha256: str, a_selected: Mapping,
                          output_dir: str | Path, *, max_states: int = 64,
                          max_rounds: int = 6) -> dict:
    """Fit Branch B against reused A attacks/decoder and bounded child updates.

    This callable is dispatched only by the central execution queue. It never
    loads outer assessment rows. A selected bank is hash-verified beforehand;
    every original attack is retained as a child-state constraint, while new
    best responses are appended in later rounds and all rho values rebased.
    """
    from experiments.pcrl_task_aligned_cuts_v1 import data, method
    from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit
    from . import alternate, fit_a, nuisance, reference

    if anchor not in (0, 1, 2) or delta not in (0., .001, .003):
        raise ValueError("registered anchor and delta required")
    if not isinstance(max_rounds, int) or not 0 <= max_rounds <= 6:
        raise ValueError("at most six registered outer updates")
    if set(role_dict) != set(fit_a.SCIENCE_ROLES):
        raise ValueError("exactly five inner roles, no outer assessment")
    inherited_audit.assert_household_disjoint(
        *(np.asarray(role_dict[name]["households"]) for name in fit_a.SCIENCE_ROLES))
    if a_selected["encoder_sha256"] != encoder_sha256:
        raise ValueError("A/B frozen encoder pins differ")
    if any(a_selected["role_input_sha256"][name] !=
           fit_a._role_fingerprint(role_dict[name])
           for name in fit_a.SCIENCE_ROLES if name != "inner_check"):
        raise ValueError("A/B original development roles differ")
    q_ref = method.validate_channel(q_ref, n_states=32, n_tokens=17)
    historical_q = method.validate_channel(historical_q, n_states=32, n_tokens=17)
    if (a_selected["q_ref_sha256"] != fit_a._sha_array(q_ref) or
            a_selected["historical_q_sha256"] != fit_a._sha_array(historical_q)):
        raise ValueError("A/B frozen comparator channels differ")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted Branch B objects require a private output directory")
    expected = {"schema": "pcrl-adaptive-B-center-v1", "anchor": anchor,
                "delta": float(delta), "max_states": max_states,
                "max_rounds": max_rounds, "support_floor_unique_households": 100,
                "support_floor_effective_households": 100,
                "encoder_sha256": encoder_sha256,
                "a_complete_receipt_sha256": a_selected["complete_receipt_sha256"],
                "a_selected_bank_sha256": a_selected["selected_bank_sha256"],
                "role_input_sha256": a_selected["role_input_sha256"],
                **_source_provenance(),
                "linux_runtime_parity": "REQUIRED_BEFORE_PRODUCTION",
                "outer_labels_accessed": False}
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        receipt = json.loads(complete_path.read_text())
        if any(receipt.get(k) != value for k, value in expected.items()):
            raise ValueError("existing Branch B center is a different scientific unit")
        if receipt.get("artifact_sha256") != fit_a._inventory(root):
            raise ValueError("completed Branch B artifact inventory differs")
        channel_path = root / receipt["selected_channel_relative"]
        with np.load(channel_path, allow_pickle=False) as archive:
            q = archive["Q"].copy()
        return {"status": receipt["status"], "selected_channel": q,
                "partition": refinement.NestedPartition.from_record(
                    json.loads((root / "PARTITION.json").read_text())),
                "complete_receipt_sha256": data.sha256_file(complete_path)}
    root.mkdir(parents=True, exist_ok=True)
    fit_a._write_or_verify_json(root / "INPUTS.json", expected)

    # The only new supervised predictor is trained on the dedicated role.
    if (root / "nuisance" / "NUISANCE.json").exists():
        nuisance_model, nuisance_receipt = nuisance.load_frozen_nuisance(root / "nuisance")
    else:
        nuisance_model, nuisance_record = nuisance.fit_frozen_nuisance(
            role_dict["nuisance_train"], seed=20260924 + 10000*anchor)
        nuisance_receipt = nuisance.save_frozen_nuisance(
            root / "nuisance", nuisance_model, nuisance_record)
    if nuisance_model.training_sha256 != nuisance_receipt["training_sha256"]:
        raise ValueError("nuisance training-role source changed")

    # A's selected fixed programme supplies real dual prices, not a success
    # string. Replay its selected decoder and cuts on the same coefficient rows.
    coefficient = role_dict["coefficient_split"]
    # Inner selection is the adaptive split-stability resource. Independent
    # common inner audit later uses inner_check, which stays out of the split.
    checking = role_dict["inner_selection"]
    a_task_pair = fit_a.task_cost_pair(a_selected["decoder"], coefficient)
    if any(np.max(np.abs(a_task_pair[w]-a_selected["cost"][w])) > 1e-9
           for w in ("U", "W")):
        raise ValueError("A selected fixed decoder cost failed independent replay")
    a_cost = 0.5*(a_task_pair["U"]+a_task_pair["W"])
    priced_lp = fixed_bank_dual_prices(a_cost, a_selected["cuts"])
    if priced_lp["status"] != "optimal" or priced_lp["numerical_gap"] > 1e-6:
        raise RuntimeError("A fixed-bank dual prices unresolved; no adaptive split")
    task_fit_mask, task_fit_loss = fit_a.task_person_losses(a_selected["decoder"], coefficient)
    task_check_mask, task_check_loss = fit_a.task_person_losses(a_selected["decoder"], checking)
    attack_fit_price, attack_fit_unique = _attack_loss_rows(
        a_selected["cuts"], a_selected["attack_specs"], coefficient)
    attack_check_price, _ = _attack_loss_rows(
        a_selected["cuts"], a_selected["attack_specs"], checking)
    g_fit = priced_original_person_rows(
        coefficient, task_fit_mask, task_fit_loss,
        attack_fit_price, priced_lp["multipliers"])
    g_check = priced_original_person_rows(
        checking, task_check_mask, task_check_loss,
        attack_check_price, priced_lp["multipliers"])
    selected = select_nested_partition(
        coefficient, checking, nuisance_model, g_fit, g_check,
        max_states=max_states, min_households=100,
        min_effective_households=100)
    partition = selected["partition"]
    fit_a._write_or_verify_json(root / "PARTITION.json", partition.to_record())
    fit_a._write_or_verify_json(root / "SPLIT_SEARCH.json", {
        "schema": 1, "status": selected["status"],
        "realized_states": len(partition.parents),
        "split_history": selected["split_history"],
        "rejected_candidates": selected["rejected_candidates"],
        "candidate_history": selected["candidate_history"],
        "split_stability_role": "inner_selection",
        "independent_inner_audit_role_untouched": "inner_check",
        "minimum_checking_to_fitting_gain_ratio": MIN_CHECK_GAIN_RATIO,
        "support_floor_unique_households": 100,
        "support_floor_effective_households": 100,
        "source_a_bank_sha256": a_selected["selected_bank_sha256"],
        "dual_prices": priced_lp["multipliers"],
        "fixed_bank_lp_objective": priced_lp["objective"],
        "fixed_bank_dual_lower_bound": priced_lp["dual_lower_bound"],
        "fixed_bank_numerical_gap": priced_lp["numerical_gap"],
        "split_gain_is_not_primal_or_audit_gain": True,
        "outer_labels_accessed": False})
    if len(partition.parents) == 32:
        channel_path = root / "channel" / "Q.npz"
        fit_a._save_or_verify_npz(channel_path, Q=a_selected["Q"])
        alias_status = ("SUPPORT_LIMITED_ALIAS_A" if selected["status"] == "support_limited"
                        else "NO_ACCEPTED_SPLIT_ALIAS_A")
        receipt = {**expected, "status": alias_status,
                   "realized_states": 32, "rounds": [],
                   "alias_reason": selected["status"],
                   "selected_channel_relative": "channel/Q.npz",
                   "a_selected_channel_array_sha256": _array_sha256(a_selected["Q"]),
                   "nuisance_model_sha256": nuisance_receipt["model_sha256"],
                   "artifact_sha256": fit_a._inventory(root)}
        fit_a._write_or_verify_json(complete_path, receipt)
        return {"status": receipt["status"], "selected_channel": a_selected["Q"].copy(),
                "partition": partition,
                "complete_receipt_sha256": data.sha256_file(complete_path)}

    child_roles = _child_roles(role_dict, partition, nuisance_model)
    n_states = len(partition.parents)
    q_ref_child = refinement.copy_parent_kernel(q_ref, partition.parent_of_leaf)
    historical_child = refinement.copy_parent_kernel(historical_q, partition.parent_of_leaf)
    current_q = refinement.copy_parent_kernel(a_selected["Q"], partition.parent_of_leaf)
    # Regroup every A attack under the new private child IDs; the new
    # coefficient-pool hash includes child routing and is shared by U/W.
    for attack in attack_fit_unique:
        valid = attack["valid_mask"]
        attack["coefficient_pool_sha256"] = fit_a._population_hash(
            child_roles["coefficient_split"], attack["target"], valid)
    child_problem = aggregate_child_problem(
        coefficient, selected["coefficient_leaves"], n_states,
        task_fit_mask, task_fit_loss, attack_fit_unique)
    retained_cuts = list(child_problem["bank"])
    retained_decoders = {"A_selected": a_selected["decoder"]}
    retained_decoder_hashes = {"A_selected": a_selected["decoder_model_sha256"]}
    rounds = []
    round_channels = []
    final_cost = None
    final_cuts = None
    for round_index in range(max_rounds + 1):
        seed = 20260924 + 10000*anchor + 100*round_index
        if round_index:
            decoder_spec = fit_a.fit_frozen_decoder(
                child_roles, current_q, q_ref_child,
                root / f"decoder_r{round_index:02d}", seed)
            did = f"B_round_{round_index:02d}/{decoder_spec['selected_candidate']}"
            retained_decoders[did] = decoder_spec["decoder"]
            retained_decoder_hashes[did] = decoder_spec["selected_model_sha256"]
            bank_dir = root / f"bank_r{round_index:02d}"
            if (bank_dir / "ATTACK_BANK.json").exists():
                fresh = fit_a.load_frozen_bank(bank_dir)
            else:
                fresh = fit_a.build_attack_bank(
                    child_roles, {f"B_candidate_r{round_index-1:02d}": current_q},
                    bank_dir, seed+50000, n_states=n_states)
            retained_cuts.extend(fresh["cuts"])
        selection_rows = child_roles["inner_selection"]
        valid = fit_a._valid_mask(selection_rows, "same_residence")
        chosen = alternate.select_frozen_decoder(
            retained_decoders,
            np.asarray(selection_rows["ha"])[valid],
            np.asarray(selection_rows["token_codes"])[valid],
            np.asarray(selection_rows["labels"]["same_residence"])[valid],
            np.asarray(selection_rows["weights"])[valid], current_q)
        task_pair = fit_a.task_cost_pair(chosen["decoder"],
                                        child_roles["coefficient_split"], n_states)
        update = alternate.channel_update(task_pair, q_ref_child,
                                          retained_cuts, delta)
        solution = update["solution"]
        if not solution.get("feasible") or solution.get("Q") is None:
            raise RuntimeError("calibrated child LP unresolved; preserve failed unit")
        q = solution["Q"]
        round_root = root / f"round_r{round_index:02d}"
        fit_a._save_or_verify_npz(round_root / "Q.npz", Q=q)
        fit_a._save_or_verify_npz(round_root / "COSTS.npz",
                                  cost_U=task_pair["U"], cost_W=task_pair["W"])
        fit_a._save_or_verify_npz(round_root / "CUTS.npz",
                                  **{f"cut_{i:04d}": c["coeff"]
                                     for i, c in enumerate(update["cuts"])})
        fit_a._write_or_verify_json(round_root / "BANK.json", {
            "schema": 1, "bank_sha256": update["bank_sha256"],
            "cut_coefficients_sha256": data.sha256_file(round_root / "CUTS.npz"),
            "cuts": [{k: v for k, v in c.items() if k != "coeff"}
                     for c in update["cuts"]],
            "rho": update["rho"]})
        selection_score = _score_inner(chosen["decoder"],
                                       child_roles["inner_selection"], q)
        round_record = {"round": round_index, "status": "FIXED_BANK_FEASIBLE",
                        "realized_states": n_states,
                        "selected_decoder_id": chosen["id"],
                        "selected_decoder_sha256": retained_decoder_hashes[chosen["id"]],
                        "bank_sha256": update["bank_sha256"],
                        "cut_count": len(retained_cuts),
                        "reference_max_cut_violation": update["witness"]["maximum_cut_violation"],
                        "candidate_max_cut_violation": solution["replay"]["maximum_cut_violation"],
                        "fixed_decoder_objective": solution["objective"],
                        "fixed_bank_dual_lower_bound": solution["dual_lower_bound"],
                        "fixed_bank_gap": solution["fixed_bank_gap"],
                        "inner_selection_fixed_decoder_task": selection_score,
                        "channel_array_sha256": _array_sha256(q),
                        "channel_file_sha256": data.sha256_file(round_root / "Q.npz"),
                        "independent_common_audit_completed": False,
                        "outer_labels_accessed": False}
        fit_a._write_or_verify_json(round_root / "ROUND.json", round_record)
        rounds.append(round_record)
        round_channels.append(q.copy())
        final_cost = 0.5*(task_pair["U"]+task_pair["W"])
        final_cuts = update["cuts"]
        if np.array_equal(q, current_q):
            break
        current_q = q
    final_selection = select_b_final_bank(round_channels, rounds,
                                           final_cost, final_cuts)
    selected_round = final_selection["selected_round"]
    selected_q = final_selection["selected_channel"]
    fit_a._write_or_verify_json(root / "FINAL_BANK_SELECTION.json", {
        key: value for key, value in final_selection.items()
        if key != "selected_channel"})
    fit_a._write_or_verify_json(root / "SELECTED.json", {
        "schema": 1, "selected_round": selected_round,
        "selection_rule": final_selection["selection_rule"],
        "selection_scope": final_selection["selection_scope"],
        "final_bank_sha256": final_selection["final_bank_sha256"],
        "final_bank_selection_sha256": data.sha256_file(root / "FINAL_BANK_SELECTION.json"),
        "selected_channel_relative": f"round_r{selected_round:02d}/Q.npz",
        "independent_common_audit_completed": False,
        "outer_labels_accessed": False})
    receipt = {**expected, "status": "COMPLETE", "realized_states": n_states,
               "rounds": rounds, "selected_round": selected_round,
               "selected_channel_relative": f"round_r{selected_round:02d}/Q.npz",
               "final_bank_sha256": final_selection["final_bank_sha256"],
               "nuisance_model_sha256": nuisance_receipt["model_sha256"],
               "partition_sha256": data.sha256_file(root / "PARTITION.json"),
               "artifact_sha256": fit_a._inventory(root)}
    fit_a._write_or_verify_json(complete_path, receipt)
    return {"status": "COMPLETE", "selected_channel": selected_q,
            "partition": partition, "selected_round": selected_round,
            "rounds": rounds,
            "complete_receipt_sha256": data.sha256_file(complete_path)}


def run_center(anchor: int, delta: float, index_path: str | Path,
               output_dir: str | Path, *, a_center_dir: str | Path,
               max_states: int = 64, max_rounds: int = 6) -> dict:
    """Hash-verified 2018 Branch B unit; requires completed same-anchor A."""
    from experiments.pcrl_task_aligned_cuts_v1 import data
    from . import fit_a, roles

    sanitized = (Path(__file__).resolve().parents[2] /
                 data.SANITIZED_RELATIVE / "SANITIZATION.json")
    if not sanitized.is_file():
        raise RuntimeError("sanitized 2018 receipt required; raw outer labels stay sealed")
    value = data.index(index_path)
    prepared = data.load_prepared(value, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise RuntimeError("outer assessment labels appeared before lock")
    role_dict = {name: roles.pooled_role(prepared, name)
                 for name in fit_a.SCIENCE_ROLES}
    q_ref = data.load_map(value, anchor, "D17")
    historical_q = data.load_map(value, anchor, "Q")
    encoder_sha = data.member_record(value, anchor, "encoder")["sha256"]
    a_selected = load_a_selected(a_center_dir, anchor=anchor, delta=delta)
    return run_center_from_roles(anchor, delta, role_dict, q_ref, historical_q,
                                 encoder_sha, a_selected, output_dir,
                                 max_states=max_states, max_rounds=max_rounds)


def verify_center_runtime_parity(anchor: int, index_path: str | Path,
                                 center_dir: str | Path,
                                 receipt_path: str | Path) -> dict:
    """Package the selected B channel only after hash-pinned Linux replay.

    The parity receipt lives outside the immutable scientific fit directory.
    A completed fit by itself is not certified as a deployable runtime.
    """
    from experiments.pcrl_task_aligned_cuts_v1 import data, release
    from experiments.pcrl_task_aligned_cuts_v1.method import validate_channel
    from . import fit_a, nuisance, roles

    root = Path(center_dir)
    target = Path(receipt_path)
    if "private" not in root.parts or "private" not in target.parts:
        raise ValueError("fit and parity receipt must remain in private paths")
    if target.resolve().is_relative_to(root.resolve()):
        raise ValueError("parity receipt must not mutate immutable fit inventory")
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("status") not in ("COMPLETE", "SUPPORT_LIMITED_ALIAS_A",
                                       "NO_ACCEPTED_SPLIT_ALIAS_A") or
            complete.get("anchor") != anchor or
            complete.get("amendment_01_sha256") != AMENDMENT_01_SHA256 or
            complete.get("artifact_sha256") != fit_a._inventory(root)):
        raise ValueError("complete B center inventory/pins failed")
    partition = refinement.NestedPartition.from_record(
        json.loads((root / "PARTITION.json").read_text()))
    with np.load(root / complete["selected_channel_relative"],
                 allow_pickle=False) as archive:
        q = archive["Q"].copy()
    q = validate_channel(q, n_states=len(partition.parents), n_tokens=17)
    frozen_nuisance, nuisance_record = nuisance.load_frozen_nuisance(root / "nuisance")
    if nuisance_record["model_sha256"] != complete["nuisance_model_sha256"]:
        raise ValueError("B fitted nuisance hash differs")
    index = data.index(index_path)
    encoder_member = data.member_record(index, anchor, "encoder")
    encoder_path = data.verified_member(encoder_member)
    if encoder_member["sha256"] != complete["encoder_sha256"]:
        raise ValueError("B frozen encoder pin differs")
    frozen_encoder = release.load_encoder_verified(
        encoder_path, encoder_member["sha256"])
    prepared = data.load_prepared(index, anchor)
    coefficient = roles.pooled_role(prepared, "coefficient_split")
    if fit_a._role_fingerprint(coefficient) != complete["role_input_sha256"]["coefficient_split"]:
        raise ValueError("coefficient parity fixture differs from fitted role")
    receipt = runtime_parity_receipt(
        coefficient, partition, frozen_encoder, frozen_nuisance,
        encoder_sha256=encoder_member["sha256"],
        nuisance_sha256=nuisance_record["model_sha256"], q=q)
    receipt.update({"center_complete_sha256": data.sha256_file(complete_path),
                    "selected_channel_relative": complete["selected_channel_relative"],
                    "selected_channel_array_sha256": _array_sha256(q),
                    "fixture_role": "coefficient_split",
                    "outer_labels_accessed": False})
    if not receipt["linux_x86_parity_verified"]:
        raise RuntimeError("Linux x86 runtime T32/child/service parity failed; no deployment receipt")
    fit_a._write_or_verify_json(target, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Private Branch B fit and runtime parity")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit/resume a bounded B scientific unit")
    fit.add_argument("--anchor", type=int, required=True)
    fit.add_argument("--delta", type=float, required=True)
    fit.add_argument("--index-path", required=True)
    fit.add_argument("--a-center-dir", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--max-states", type=int, default=64)
    fit.add_argument("--max-rounds", type=int, default=6)
    parity = commands.add_parser("parity", help="verify selected runtime on Linux x86")
    parity.add_argument("--anchor", type=int, required=True)
    parity.add_argument("--index-path", required=True)
    parity.add_argument("--center-dir", required=True)
    parity.add_argument("--receipt-path", required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        result = run_center(args.anchor, args.delta, args.index_path,
                            args.output_dir, a_center_dir=args.a_center_dir,
                            max_states=args.max_states, max_rounds=args.max_rounds)
        print(json.dumps({"status": result["status"],
                          "realized_states": len(result["partition"].parents),
                          "complete_receipt_sha256": result["complete_receipt_sha256"],
                          "linux_runtime_parity": "REQUIRED_BEFORE_PRODUCTION"},
                         sort_keys=True))
    else:
        receipt = verify_center_runtime_parity(
            args.anchor, args.index_path, args.center_dir, args.receipt_path)
        print(json.dumps({"linux_x86_parity_verified": receipt["linux_x86_parity_verified"],
                          "selected_channel_array_sha256": receipt["selected_channel_array_sha256"],
                          "checked_original_people": receipt["checked_original_people"]},
                         sort_keys=True))


if __name__ == "__main__":
    main()
