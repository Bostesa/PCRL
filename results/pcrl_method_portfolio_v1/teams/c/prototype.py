"""Team C's fixed grouped-mask release boundary; no data loading or fitting.

The schema follows the pinned JCV Adult input builder. One global mask is used
per purpose. Each head receives only the selected raw values. The calibrator
receives only that head's output. The complete release exposes selected raw
values, calibrated probabilities, decisions, and public identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


NUMERIC = (
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)
CATEGORICAL = {
    "workclass": (
        "Federal-gov", "Local-gov", "Never-worked", "Private",
        "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
    ),
    "education": tuple(sorted((
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Masters", "Doctorate", "Prof-school",
    ))),
    "marital-status": tuple(sorted((
        "Never-married", "Divorced", "Separated", "Widowed",
        "Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse",
    ))),
    "relationship": (
        "Husband", "Not-in-family", "Other-relative", "Own-child",
        "Unmarried", "Wife",
    ),
    "native-country": (
        "Cambodia", "Canada", "China", "Columbia", "Cuba",
        "Dominican-Republic", "Ecuador", "El-Salvador", "England",
        "France", "Germany", "Greece", "Guatemala", "Haiti",
        "Holand-Netherlands", "Honduras", "Hong", "Hungary",
        "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan",
        "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)",
        "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico",
        "Scotland", "South", "Taiwan", "Thailand",
        "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia",
    ),
}
FEATURE_NAMES = NUMERIC + tuple(
    f"{group}={category}"
    for group, categories in CATEGORICAL.items()
    for category in categories
)
ATOM_ORDER = (
    "age", "education_fact", "capital-gain", "capital-loss",
    "hours-per-week", "workclass", "marital-status", "relationship",
    "native-country",
)


class ProbabilityHead(Protocol):
    def predict_proba(self, selected: np.ndarray) -> np.ndarray: ...


class ProbabilityCalibrator(Protocol):
    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray: ...


class IdentityCalibrator:
    """Explicit no-op calibrator for synthetic checks and uncalibrated controls."""

    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(raw_probabilities).copy()


class AdultGroupedSchema:
    """Exact 83-column JCV schema with nine coherent selection atoms."""

    feature_names = FEATURE_NAMES

    def __init__(self) -> None:
        index = {name: i for i, name in enumerate(self.feature_names)}
        self.atoms = {
            "age": (index["age"],),
            "education_fact": (index["education-num"],) + tuple(
                index[name] for name in self.feature_names
                if name.startswith("education=")
            ),
            "capital-gain": (index["capital-gain"],),
            "capital-loss": (index["capital-loss"],),
            "hours-per-week": (index["hours-per-week"],),
            "workclass": tuple(
                index[name] for name in self.feature_names
                if name.startswith("workclass=")
            ),
            "marital-status": tuple(
                index[name] for name in self.feature_names
                if name.startswith("marital-status=")
            ),
            "relationship": tuple(
                index[name] for name in self.feature_names
                if name.startswith("relationship=")
            ),
            "native-country": tuple(
                index[name] for name in self.feature_names
                if name.startswith("native-country=")
            ),
        }

    def validate_feature_names(self, feature_names: tuple[str, ...]) -> None:
        if tuple(feature_names) != self.feature_names:
            raise ValueError("input does not match the pinned ordered 83-column schema")

    def normalize_atoms(self, selected_atoms: tuple[str, ...]) -> tuple[str, ...]:
        selected_atoms = tuple(selected_atoms)
        unknown = set(selected_atoms) - set(self.atoms)
        if unknown:
            raise ValueError(f"unknown selection atom: {sorted(unknown)}")
        if len(selected_atoms) != len(set(selected_atoms)):
            raise ValueError("duplicate selection atom")
        return tuple(atom for atom in ATOM_ORDER if atom in selected_atoms)

    def column_indices(self, selected_atoms: tuple[str, ...]) -> tuple[int, ...]:
        selected = set(self.normalize_atoms(selected_atoms))
        return tuple(sorted(i for atom in selected for i in self.atoms[atom]))


@dataclass(frozen=True)
class PurposeRelease:
    schema: AdultGroupedSchema
    selected_atoms: tuple[str, ...]
    head: ProbabilityHead
    purpose: str
    head_id: str
    calibrator_id: str
    calibrator: ProbabilityCalibrator

    def __post_init__(self) -> None:
        if self.purpose not in ("income", "occupation_group"):
            raise ValueError("purpose must be income or occupation_group")
        if not self.head_id or not self.calibrator_id:
            raise ValueError("head and calibrator IDs must be explicit")
        object.__setattr__(
            self, "selected_atoms", self.schema.normalize_atoms(self.selected_atoms)
        )

    def view(self, X: np.ndarray, feature_names: tuple[str, ...]) -> dict:
        self.schema.validate_feature_names(feature_names)
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 83 or not np.isfinite(X).all():
            raise ValueError("X must be a finite two-dimensional 83-column matrix")
        indices = self.schema.column_indices(self.selected_atoms)
        selected = X[:, indices].copy()
        raw_probabilities = np.asarray(
            self.head.predict_proba(selected.copy()), dtype=float
        )
        probabilities = np.asarray(
            self.calibrator.transform(raw_probabilities.copy()), dtype=float
        )
        expected_classes = 2 if self.purpose == "income" else 6
        if probabilities.ndim != 2 or probabilities.shape[1] != expected_classes:
            label = "two class" if expected_classes == 2 else "six class"
            raise ValueError(f"{self.purpose} requires a {label} probability vector")
        if (
            probabilities.shape[0] != X.shape[0]
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)
        ):
            raise ValueError("head returned invalid class probabilities")
        return {
            "selected_values": selected,
            "feature_names": tuple(self.schema.feature_names[i] for i in indices),
            "selected_groups": self.selected_atoms,
            "mask_id": "+".join(self.selected_atoms) or "none",
            "head_id": self.head_id,
            "calibrator_id": self.calibrator_id,
            "flags": (),
            "probabilities": probabilities,
            "decisions": probabilities.argmax(axis=1),
        }


def candidate_pair(
    schema: AdultGroupedSchema,
    income_atoms: tuple[str, ...],
    occupation_atoms: tuple[str, ...],
    income_head: ProbabilityHead,
    occupation_head: ProbabilityHead,
    income_head_id: str,
    occupation_head_id: str,
    calibrator_id: str,
    income_calibrator: ProbabilityCalibrator,
    occupation_calibrator: ProbabilityCalibrator,
) -> tuple[PurposeRelease, PurposeRelease]:
    """Construct the predeclared proxy-excluding Team C candidate interface."""
    if "relationship" in income_atoms or "relationship" in occupation_atoms:
        raise ValueError("relationship is excluded from Team C candidate masks")
    return (
        PurposeRelease(
            schema, income_atoms, income_head, "income", income_head_id,
            calibrator_id, income_calibrator,
        ),
        PurposeRelease(
            schema, occupation_atoms, occupation_head, "occupation_group",
            occupation_head_id, calibrator_id, occupation_calibrator,
        ),
    )


def release_pair(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    income: PurposeRelease,
    occupation: PurposeRelease,
) -> dict:
    """Return both complete local views and the aligned coalition view."""
    if income.purpose != "income" or occupation.purpose != "occupation_group":
        raise ValueError("release pair requires income and occupation_group in order")
    if income.schema.feature_names != occupation.schema.feature_names:
        raise ValueError("recipient schemas differ")
    first = income.view(X, feature_names)
    second = occupation.view(X, feature_names)
    schema = income.schema
    union_atoms = schema.normalize_atoms(
        tuple(set(income.selected_atoms) | set(occupation.selected_atoms))
    )
    indices = schema.column_indices(union_atoms)
    coalition = {
        "selected_values": np.asarray(X)[:, indices].copy(),
        "feature_names": tuple(schema.feature_names[i] for i in indices),
        "selected_groups": union_atoms,
        "income_groups": income.selected_atoms,
        "occupation_group_groups": occupation.selected_atoms,
        "income_mask_id": first["mask_id"],
        "occupation_group_mask_id": second["mask_id"],
        "income_head_id": income.head_id,
        "occupation_group_head_id": occupation.head_id,
        "income_calibrator_id": income.calibrator_id,
        "occupation_group_calibrator_id": occupation.calibrator_id,
        "flags": (),
        "income_probabilities": first["probabilities"],
        "occupation_group_probabilities": second["probabilities"],
        "income_decisions": first["decisions"],
        "occupation_group_decisions": second["decisions"],
    }
    return {"income": first, "occupation_group": second, "coalition": coalition}
