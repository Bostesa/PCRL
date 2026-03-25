"""PurposeSpec dataclass and purpose registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PurposeSpec:
    """Specification for a data processing purpose.

    Defines what tasks are allowed and what sensitive attributes must be
    hidden for a specific data processing purpose.

    Attributes:
        name: Human-readable purpose identifier (e.g., "income_prediction").
        allowed_tasks: Task names this purpose permits (e.g., ["income"]).
        disallowed_attrs: Sensitive attributes that must not be predictable.
        task_type: Type of prediction task ("classification" or "regression").
        allowed_task_dims: Output dimensionality per allowed task.
        disallowed_attr_dims: Number of classes per disallowed attribute.
    """

    name: str
    allowed_tasks: list[str]
    disallowed_attrs: list[str]
    task_type: Literal["classification", "regression"] = "classification"
    allowed_task_dims: dict[str, int] = field(default_factory=dict)
    disallowed_attr_dims: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.allowed_tasks:
            raise ValueError(
                f"Purpose '{self.name}' must have at least one allowed task"
            )
        if not self.disallowed_attrs:
            raise ValueError(
                f"Purpose '{self.name}' must have at least one disallowed attribute"
            )
        if self.task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got {self.task_type}"
            )


class PurposeRegistry:
    """Registry for managing purpose specifications.

    Provides registration, lookup, and validation of purposes.
    """

    def __init__(self) -> None:
        self._purposes: dict[str, PurposeSpec] = {}

    def register(self, purpose_spec: PurposeSpec) -> None:
        """Register a purpose specification.

        Args:
            purpose_spec: The purpose to register.

        Raises:
            ValueError: If a purpose with the same name is already registered.
        """
        if purpose_spec.name in self._purposes:
            raise ValueError(
                f"Purpose '{purpose_spec.name}' is already registered"
            )
        self._purposes[purpose_spec.name] = purpose_spec

    def get(self, name: str) -> PurposeSpec:
        """Retrieve a purpose specification by name.

        Args:
            name: The purpose name.

        Returns:
            The corresponding PurposeSpec.

        Raises:
            KeyError: If no purpose with this name is registered.
        """
        if name not in self._purposes:
            raise KeyError(
                f"Purpose '{name}' not found. "
                f"Registered purposes: {list(self._purposes.keys())}"
            )
        return self._purposes[name]

    def all_disallowed_for(self, purpose_name: str) -> set[str]:
        """Return the set of all disallowed attributes for a given purpose.

        Args:
            purpose_name: Name of the purpose.

        Returns:
            Set of disallowed attribute names.
        """
        purpose = self.get(purpose_name)
        return set(purpose.disallowed_attrs)

    @property
    def names(self) -> list[str]:
        """Return list of all registered purpose names."""
        return list(self._purposes.keys())

    @property
    def purposes(self) -> list[PurposeSpec]:
        """Return list of all registered purpose specs."""
        return list(self._purposes.values())

    def __len__(self) -> int:
        return len(self._purposes)

    def __contains__(self, name: str) -> bool:
        return name in self._purposes

    def __iter__(self):
        return iter(self._purposes.values())

    def __repr__(self) -> str:
        return f"PurposeRegistry(purposes={self.names})"
