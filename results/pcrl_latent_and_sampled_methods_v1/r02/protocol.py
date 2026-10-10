"""Shared fixed attack and task endpoint contract for D and A releases.

No data loader or fit is invoked here. Numerical workers must first verify the
R00 effective lock and hold R03's global heavy-process slot. The inherited
``attack_eval`` module does the group split, ATT fit/TUNE selection, zero-fit
ignore-recipient aliases, private reader custody and INNER scoring.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import joblib
import numpy as np

_SHARED = Path(__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from attack_eval import (_factory, _label_record, add_external_fixed_readers,
                         add_ignore_recipient_aliases, attack_fit_tune_split,
                         fit_slate, save_fitted_slate, score_slate)
from attack_controls_v2 import run_control_probes_v2, save_control_probe_v2_custody
from common_eval import assert_same_people, task_metrics, utility_against_references
from safe_data_guard import sha_file

from .complete_view import PreparedViews


@dataclass(frozen=True)
class ReaderPlan:
    family: str
    core: tuple[str, ...]
    probability: tuple[str, ...]
    views: tuple[str, ...] = ("r1", "r2", "pair")
    split: str = "inherited_sha_3046_train_1015_tune_groups"
    orientation: str = "TUNE_positive_or_reverse; INNER_fixed_alias_max_envelope"


def reader_plan(*, family: str, code_bits: bool) -> ReaderPlan:
    """The mandatory common slate; family specialists are separate locked IDs."""
    if family not in ("D", "A") or not code_bits:
        raise ValueError("family and bit-reader applicability disagree")
    core = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32", "CELL_A1")
    if code_bits:
        core += ("CODE_BIT_LR_C1",)
    return ReaderPlan(family, core, ("PROB_LR_C1", "PROB_HGB_15x200"))


def unit_fit_forecast(plan: ReaderPlan) -> dict:
    """Exact count per distinct release/seed with three fresh views and V2 controls."""
    if plan.views != ("r1", "r2", "pair"):
        raise ValueError("three local/pair views required")
    core = 3 * len(plan.core)
    probability = 3 * len(plan.probability)
    controls = 2 * 3  # V2 shuffled-label LR and planted-leak LR in each view
    return {"core_fits": core, "probability_only_fits": probability,
            "v2_control_fits": controls, "ignore_recipient_fits": 0,
            "mandatory_fits": core + probability + controls,
            "views": plan.views,
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "tune_role": "ATTACK_FIT_NEW/ATTACK_TUNE",
            "score_role": "INNER_SELECTION"}


def score_release_tasks(candidate: Mapping, truth: Mapping, references: Mapping[str, Mapping],
                        *, constant_prob: Mapping[str, np.ndarray] | None = None) -> dict:
    """Identically ordered, two-task point endpoints against both U0 and Ucal."""
    if set(references) != {"U0", "Ucal"}:
        raise ValueError("both original reference identities are required")
    try:
        assert_same_people(candidate, truth, *references.values())
    except (KeyError, ValueError) as exc:
        raise ValueError("ordered row/group identity differs") from exc
    result = {}
    for task in ("income", "occupation"):
        cp = None if constant_prob is None else constant_prob.get(task)
        metrics = task_metrics(candidate[task], truth[task], task=task, constant_prob=cp)
        ref = {name: task_metrics(references[name][task], truth[task], task=task)
               for name in ("U0", "Ucal")}
        comparison = utility_against_references(metrics, ref["U0"], ref["Ucal"])
        result[task] = {"metrics": metrics, "references": {name: {
            "metrics": ref[name], **comparison[name]} for name in ("U0", "Ucal")},
            "point_eligible_both": comparison["point_eligible_both"]}
    result["point_eligible_both_tasks"] = all(result[t]["point_eligible_both"]
                                              for t in ("income", "occupation"))
    return result


def score_three_view_bank(fitted: Mapping[str, dict], sex_inner: dict) -> dict:
    """Score frozen local and pair predictions, with zero-fit local aliases in pair."""
    if set(fitted) != {"r1", "r2", "pair"}:
        raise ValueError("all three fitted complete views are required")
    pair = add_ignore_recipient_aliases(fitted["pair"], fitted["r1"], fitted["r2"])
    return {"r1": score_slate(fitted["r1"], sex_inner),
            "r2": score_slate(fitted["r2"], sex_inner),
            "pair": score_slate(pair, sex_inner)}


def fit_probability_specialists(
    prepared: PreparedViews, sex_att: dict, *, seed: int, private_dir: str | Path,
    lock_path: str | Path, lock_sha256: str, queue_guard: Callable[[], None],
) -> tuple[dict, dict]:
    """Fit two fixed output-only readers for one view under caller's active slot.

    Return records for inherited ``add_external_fixed_readers``. The caller
    supplies a queue guard that raises unless R03's slot is active; this helper
    checks it before each fit. No INNER label is accepted. Do not call before
    R00 seals the effective lock and R03 admits the numerical job.
    """
    lock = Path(lock_path)
    if len(lock_sha256) != 64 or not lock.is_file() or sha_file(lock) != lock_sha256:
        raise ValueError("effective lock file/hash missing or changed")
    if not callable(queue_guard):
        raise ValueError("active queue guard required")
    out = Path(private_dir)
    if ".private" not in out.resolve().parts or out.exists():
        raise ValueError("probability reader custody requires a new private directory")
    if not isinstance(prepared, PreparedViews):
        raise ValueError("prepared complete views are required")
    prepared.att.validate()
    prepared.inner.validate()
    pa = np.asarray(prepared.probability_att, dtype=np.float64)
    pi = np.asarray(prepared.probability_inner, dtype=np.float64)
    if pa.ndim != 2 or pi.ndim != 2 or pa.shape[0] != len(prepared.att.row_id) or \
            pi.shape[0] != len(prepared.inner.row_id) or pa.shape[1] == 0 or pa.shape[1] != pi.shape[1] or \
            not np.isfinite(pa).all() or not np.isfinite(pi).all():
        raise ValueError("matched output probability matrices are required")
    y = _label_record(sex_att, prepared.att)
    split = attack_fit_tune_split(prepared.att.unit)
    fit, tune = split["fit"], split["tune"]
    if np.unique(y[fit]).size != 2 or np.unique(y[tune]).size != 2:
        raise ValueError("ATT fit/tune needs both SEX classes")
    queue_guard()
    out.mkdir(parents=True, mode=0o700)
    result = []
    for name, base in (("PROB_LR_C1", "LR_C1"),
                       ("PROB_HGB_15x200", "HGB_15x200")):
        queue_guard()
        model = _factory(base, seed)
        model.fit(pa[fit], y[fit])
        path = out / f"model__{name}.joblib"
        joblib.dump(model, path, compress=3)
        path.chmod(0o600)
        result.append({"name": name, "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
                       "model_artifact_path": str(path), "model_sha256": sha_file(path),
                       "prediction_tune": np.asarray(model.predict_proba(pa[tune])[:, 1], dtype=np.float64),
                       "prediction_inner": np.asarray(model.predict_proba(pi)[:, 1], dtype=np.float64)})
    return tuple(result)


def run_three_view_attack(
    prepared: Mapping[str, PreparedViews], sex_att: dict, sex_inner: dict, *,
    family: str, seed: int, private_root: str | Path,
    lock_path: str | Path, lock_sha256: str, queue_guard: Callable[[], None],
    specialists_by_view: Mapping[str, tuple[dict, ...]] | None = None,
) -> dict:
    """Fit/custody/score the frozen three-view slate and V2 controls.

    This is the numerical worker called *inside* R03's slot. The caller first
    verifies role admission and release identity. The return's ``fitted`` bank
    contains private models/predictions for paired inference; only ``public``
    is JSON-safe. The pair bank includes zero-fit local reader aliases.
    """
    lock = Path(lock_path)
    if len(lock_sha256) != 64 or not lock.is_file() or sha_file(lock) != lock_sha256:
        raise ValueError("effective lock file/hash missing or changed")
    if not callable(queue_guard):
        raise ValueError("active queue guard required")
    if set(prepared) != {"r1", "r2", "pair"}:
        raise ValueError("all three prepared complete views are required")
    if specialists_by_view is not None and set(specialists_by_view) != set(prepared):
        raise ValueError("predeclared specialist coverage must include all views")
    root = Path(private_root)
    if ".private" not in root.resolve().parts or root.exists():
        raise ValueError("attack custody requires a new private directory")
    plan = reader_plan(family=family, code_bits=True)
    for side in ("att", "inner"):
        anchor = getattr(prepared["r1"], side)
        for name in ("r2", "pair"):
            view = getattr(prepared[name], side)
            if not np.array_equal(anchor.row_id, view.row_id) or not np.array_equal(anchor.unit, view.unit):
                raise ValueError("ordered people differ across complete views")
    queue_guard()
    root.mkdir(parents=True, mode=0o700)
    fitted = {}
    for name in plan.views:
        view = prepared[name]
        queue_guard()
        core = fit_slate(view.att, sex_att, view.inner,
                         required_readers=plan.core, attacker_seed=seed)
        specialists = fit_probability_specialists(
            view, sex_att, seed=seed, private_dir=root / name / "probability",
            lock_path=lock, lock_sha256=lock_sha256, queue_guard=queue_guard)
        extras = tuple(specialists_by_view[name]) if specialists_by_view is not None else ()
        fitted[name] = add_external_fixed_readers(core, view.att, sex_att,
                                                   view.inner, specialists + extras)
    fitted["pair"] = add_ignore_recipient_aliases(fitted["pair"], fitted["r1"], fitted["r2"])
    scores, custody, controls = {}, {}, {}
    for name in plan.views:
        view = prepared[name]
        scores[name] = score_slate(fitted[name], sex_inner)
        custody[name] = save_fitted_slate(fitted[name], root / name / "readers",
                                          att_view=view.att, inner_view=view.inner)
        queue_guard()
        control, artifacts = run_control_probes_v2(
            view.att, sex_att, view.inner, sex_inner, return_artifacts=True)
        controls[name] = {"scores": control,
                          "custody": save_control_probe_v2_custody(
                              artifacts, root / name / "controls")}
    public = {"schema": "pcrl-next-frozen-attack-unit-v1", "family": family,
              "model_seed": seed, "effective_lock_sha256": lock_sha256,
              "plan": {"core": plan.core, "probability": plan.probability,
                       "fit_forecast": unit_fit_forecast(plan)},
              "scores": scores, "reader_custody": custody, "controls": controls,
              "view_schema_sha256": {name: prepared[name].schema["schema_sha256"]
                                     for name in plan.views}}
    return {"public": public, "fitted": fitted}
