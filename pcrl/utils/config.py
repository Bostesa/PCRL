"""Configuration dataclasses for PCRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from pcrl.purposes.spec import PurposeRegistry, PurposeSpec


@dataclass
class PCRLConfig:
    """Configuration for the Purpose-Conditioned Representation Learning model.

    Attributes:
        input_dim: Dimensionality of input features.
        hidden_dims: List of hidden layer dimensions for the encoder.
        repr_dim: Dimensionality of the learned representation.
        purpose_emb_dim: Dimensionality of purpose embeddings.
        num_purposes: Number of distinct purposes.
        conditioning: Method for conditioning on purpose ("film", "concat", or "attention").
    """

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    repr_dim: int = 128
    purpose_emb_dim: int = 64
    num_purposes: int = 1
    conditioning: Literal["film", "concat", "attention"] = "film"

    def __post_init__(self) -> None:
        if self.conditioning not in ("film", "concat", "attention"):
            raise ValueError(
                f"conditioning must be 'film', 'concat', or 'attention', "
                f"got {self.conditioning}"
            )
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.repr_dim <= 0:
            raise ValueError(f"repr_dim must be positive, got {self.repr_dim}")
        if self.num_purposes <= 0:
            raise ValueError(
                f"num_purposes must be positive, got {self.num_purposes}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training the PCRL model.

    Attributes:
        batch_size: Number of samples per training batch.
        lr_encoder: Learning rate for the encoder.
        lr_task: Learning rate for task heads.
        lr_auditor: Learning rate for adversarial auditors.
        lambda_adv: Weight for the adversarial loss term.
        lambda_verify: Weight for verification regularizer (0 to disable).
        auditor_steps: Number of auditor updates per encoder update.
        epochs: Total number of training epochs.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm for clipping (None to disable).
        early_stopping_patience: Epochs without improvement before stopping.
        checkpoint_dir: Directory to save model checkpoints.
        log_interval: Steps between logging updates.
        eval_interval: Steps between evaluation runs.
        seed: Random seed for reproducibility.
    """

    batch_size: int = 256
    lr_encoder: float = 1e-3
    lr_task: float = 1e-3
    lr_auditor: float = 1e-3
    lambda_adv: float = 1.0
    lambda_verify: float = 0.0
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
        if self.lr_encoder <= 0 or self.lr_task <= 0 or self.lr_auditor <= 0:
            raise ValueError("Learning rates must be positive")
        if self.lambda_adv < 0:
            raise ValueError(
                f"lambda_adv must be non-negative, got {self.lambda_adv}"
            )
        if self.auditor_steps <= 0:
            raise ValueError(
                f"auditor_steps must be positive, got {self.auditor_steps}"
            )


@dataclass
class ExperimentConfig:
    """Complete configuration for a PCRL experiment.

    Attributes:
        model: Model configuration.
        training: Training configuration.
        purposes: List of purpose specifications.
        data_dir: Directory containing datasets.
        output_dir: Directory for experiment outputs.
        experiment_name: Name for this experiment run.
    """

    model: PCRLConfig
    training: TrainingConfig
    purposes: list[PurposeSpec] = field(default_factory=list)
    data_dir: str = "data"
    output_dir: str = "outputs"
    experiment_name: str = "pcrl_experiment"

    def build_registry(self) -> PurposeRegistry:
        """Build a PurposeRegistry from the configured purposes."""
        registry = PurposeRegistry()
        for p in self.purposes:
            registry.register(p)
        return registry


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed ExperimentConfig object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return _parse_experiment_config(raw)


# Backward-compatible alias
load_config_from_yaml = load_config


def _parse_experiment_config(raw: dict) -> ExperimentConfig:
    """Parse a raw dictionary into an ExperimentConfig."""
    purposes: list[PurposeSpec] = []
    for p in raw.get("model", {}).get("purposes", []):
        purposes.append(
            PurposeSpec(
                name=p["name"],
                allowed_tasks=p["allowed_tasks"],
                disallowed_attrs=p["disallowed_attrs"],
                task_type=p.get("task_type", "classification"),
                allowed_task_dims=p.get("allowed_task_dims", {}),
                disallowed_attr_dims=p.get("disallowed_attr_dims", {}),
            )
        )

    model_raw = raw.get("model", {})
    model_config = PCRLConfig(
        input_dim=model_raw.get("input_dim", 64),
        hidden_dims=model_raw.get("hidden_dims", [256, 256]),
        repr_dim=model_raw.get("repr_dim", 128),
        purpose_emb_dim=model_raw.get("purpose_emb_dim", 64),
        num_purposes=len(purposes),
        conditioning=model_raw.get("conditioning", "film"),
    )

    training_raw = raw.get("training", {})
    training_config = TrainingConfig(
        batch_size=training_raw.get("batch_size", 256),
        lr_encoder=training_raw.get("lr_encoder", 1e-3),
        lr_task=training_raw.get("lr_task", 1e-3),
        lr_auditor=training_raw.get("lr_auditor", 1e-3),
        lambda_adv=training_raw.get("lambda_adv", 1.0),
        lambda_verify=training_raw.get("lambda_verify", 0.0),
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
        purposes=purposes,
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
                    "allowed_task_dims": p.allowed_task_dims,
                    "disallowed_attr_dims": p.disallowed_attr_dims,
                }
                for p in config.purposes
            ],
        },
        "training": {
            "batch_size": config.training.batch_size,
            "lr_encoder": config.training.lr_encoder,
            "lr_task": config.training.lr_task,
            "lr_auditor": config.training.lr_auditor,
            "lambda_adv": config.training.lambda_adv,
            "lambda_verify": config.training.lambda_verify,
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
