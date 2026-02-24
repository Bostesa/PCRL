"""Configuration dataclasses for PCRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PurposeSpec:
    """Specification for a data processing purpose.

    Attributes:
        name: Human-readable purpose identifier (e.g., "income_prediction").
        allowed_tasks: List of task names this purpose permits (e.g., ["income"]).
        disallowed_attrs: Sensitive attributes that must not be predictable (e.g., ["race", "sex"]).
        task_type: Type of prediction task ("classification" or "regression").
    """
    name: str
    allowed_tasks: list[str]
    disallowed_attrs: list[str]
    task_type: Literal["classification", "regression"] = "classification"

    def __post_init__(self) -> None:
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {self.task_type}")


@dataclass
class PCRLConfig:
    """Configuration for the Purpose-Conditioned Representation Learning model.

    Attributes:
        input_dim: Dimensionality of input features.
        hidden_dims: List of hidden layer dimensions for the encoder.
        repr_dim: Dimensionality of the learned representation.
        purpose_emb_dim: Dimensionality of purpose embeddings.
        purposes: List of purpose specifications.
        conditioning: Method for conditioning on purpose ("film", "concat", or "attention").
    """
    input_dim: int
    hidden_dims: list[int]
    repr_dim: int
    purpose_emb_dim: int
    purposes: list[PurposeSpec]
    conditioning: Literal["film", "concat", "attention"] = "film"

    def __post_init__(self) -> None:
        if self.conditioning not in ("film", "concat", "attention"):
            raise ValueError(f"conditioning must be 'film', 'concat', or 'attention', got {self.conditioning}")
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.repr_dim <= 0:
            raise ValueError(f"repr_dim must be positive, got {self.repr_dim}")

    @property
    def num_purposes(self) -> int:
        """Return the number of purposes."""
        return len(self.purposes)

    def get_purpose_by_name(self, name: str) -> PurposeSpec | None:
        """Get a purpose specification by name."""
        for purpose in self.purposes:
            if purpose.name == name:
                return purpose
        return None


@dataclass
class TrainingConfig:
    """Configuration for training the PCRL model.

    Attributes:
        batch_size: Number of samples per training batch.
        lr_encoder: Learning rate for the encoder and task heads.
        lr_auditor: Learning rate for adversarial auditors.
        lambda_adv: Weight for the adversarial loss term.
        auditor_steps: Number of auditor updates per encoder update.
        epochs: Total number of training epochs.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm for clipping (None to disable).
        early_stopping_patience: Epochs without improvement before stopping (None to disable).
        checkpoint_dir: Directory to save model checkpoints.
        log_interval: Steps between logging updates.
        eval_interval: Steps between evaluation runs.
        seed: Random seed for reproducibility.
    """
    batch_size: int = 256
    lr_encoder: float = 1e-3
    lr_auditor: float = 1e-3
    lambda_adv: float = 1.0
    auditor_steps: int = 5
    epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    early_stopping_patience: int | None = 10
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    eval_interval: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr_encoder <= 0 or self.lr_auditor <= 0:
            raise ValueError("Learning rates must be positive")
        if self.lambda_adv < 0:
            raise ValueError(f"lambda_adv must be non-negative, got {self.lambda_adv}")
        if self.auditor_steps <= 0:
            raise ValueError(f"auditor_steps must be positive, got {self.auditor_steps}")


@dataclass
class ExperimentConfig:
    """Complete configuration for a PCRL experiment.

    Attributes:
        model: Model configuration.
        training: Training configuration.
        data_dir: Directory containing datasets.
        output_dir: Directory for experiment outputs.
        experiment_name: Name for this experiment run.
    """
    model: PCRLConfig
    training: TrainingConfig
    data_dir: str = "data"
    output_dir: str = "outputs"
    experiment_name: str = "pcrl_experiment"


def load_config_from_yaml(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed ExperimentConfig object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return _parse_experiment_config(raw_config)


def _parse_experiment_config(raw: dict) -> ExperimentConfig:
    """Parse a raw dictionary into an ExperimentConfig."""
    # Parse purposes
    purposes = []
    for p in raw.get("model", {}).get("purposes", []):
        purposes.append(PurposeSpec(
            name=p["name"],
            allowed_tasks=p["allowed_tasks"],
            disallowed_attrs=p["disallowed_attrs"],
            task_type=p.get("task_type", "classification"),
        ))

    # Parse model config
    model_raw = raw.get("model", {})
    model_config = PCRLConfig(
        input_dim=model_raw.get("input_dim", 64),
        hidden_dims=model_raw.get("hidden_dims", [256, 128]),
        repr_dim=model_raw.get("repr_dim", 64),
        purpose_emb_dim=model_raw.get("purpose_emb_dim", 32),
        purposes=purposes,
        conditioning=model_raw.get("conditioning", "film"),
    )

    # Parse training config
    training_raw = raw.get("training", {})
    training_config = TrainingConfig(
        batch_size=training_raw.get("batch_size", 256),
        lr_encoder=training_raw.get("lr_encoder", 1e-3),
        lr_auditor=training_raw.get("lr_auditor", 1e-3),
        lambda_adv=training_raw.get("lambda_adv", 1.0),
        auditor_steps=training_raw.get("auditor_steps", 5),
        epochs=training_raw.get("epochs", 100),
        weight_decay=training_raw.get("weight_decay", 1e-4),
        grad_clip=training_raw.get("grad_clip", 1.0),
        early_stopping_patience=training_raw.get("early_stopping_patience", 10),
        checkpoint_dir=training_raw.get("checkpoint_dir", "checkpoints"),
        log_interval=training_raw.get("log_interval", 100),
        eval_interval=training_raw.get("eval_interval", 500),
        seed=training_raw.get("seed", 42),
    )

    return ExperimentConfig(
        model=model_config,
        training=training_config,
        data_dir=raw.get("data_dir", "data"),
        output_dir=raw.get("output_dir", "outputs"),
        experiment_name=raw.get("experiment_name", "pcrl_experiment"),
    )


def save_config_to_yaml(config: ExperimentConfig, path: str | Path) -> None:
    """Save an experiment configuration to a YAML file.

    Args:
        config: The configuration to save.
        path: Path where the YAML file will be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary
    raw = {
        "experiment_name": config.experiment_name,
        "data_dir": config.data_dir,
        "output_dir": config.output_dir,
        "model": {
            "input_dim": config.model.input_dim,
            "hidden_dims": config.model.hidden_dims,
            "repr_dim": config.model.repr_dim,
            "purpose_emb_dim": config.model.purpose_emb_dim,
            "conditioning": config.model.conditioning,
            "purposes": [
                {
                    "name": p.name,
                    "allowed_tasks": p.allowed_tasks,
                    "disallowed_attrs": p.disallowed_attrs,
                    "task_type": p.task_type,
                }
                for p in config.model.purposes
            ],
        },
        "training": {
            "batch_size": config.training.batch_size,
            "lr_encoder": config.training.lr_encoder,
            "lr_auditor": config.training.lr_auditor,
            "lambda_adv": config.training.lambda_adv,
            "auditor_steps": config.training.auditor_steps,
            "epochs": config.training.epochs,
            "weight_decay": config.training.weight_decay,
            "grad_clip": config.training.grad_clip,
            "early_stopping_patience": config.training.early_stopping_patience,
            "checkpoint_dir": config.training.checkpoint_dir,
            "log_interval": config.training.log_interval,
            "eval_interval": config.training.eval_interval,
            "seed": config.training.seed,
        },
    }

    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
