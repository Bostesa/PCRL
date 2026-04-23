# PCRL: Purpose-Conditioned Representation Learning

A PyTorch framework for learning fair representations conditioned on declared data-processing purposes. A single encoder produces different representations depending on the stated purpose, filtering out sensitive attributes that are disallowed for that purpose while preserving task-relevant information.

## Overview

PCRL addresses the problem of purpose limitation in data processing: different uses of the same data may require access to different information, and sensitive attributes that are irrelevant to a given purpose should not be recoverable from the representation. The encoder is trained adversarially — auditors try to recover disallowed attributes, and the encoder learns to suppress them.

### Key Features

- **Purpose-Conditioned Encoding**: A single encoder produces purpose-specific representations via FiLM, concatenation, or cross-attention conditioning
- **Adversarial Training**: Multi-attribute auditors attempt to recover disallowed sensitive attributes; the encoder is trained to confuse them
- **Compliance Certificates**: Linear R² certificates (closed-form) and empirical post-hoc audits (LogisticRegression, RandomForest, SVM, XGBoost) to verify attribute suppression
- **Multiple Datasets**: Adult/Census (tabular), HAR (sensor/time-series), CelebA (vision)
- **Baselines**: Standard (no protection), LAFTR (single shared adversarial representation), INLP, LEACE

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+.

## Project Structure

```
pcrl/
├── pcrl/
│   ├── models/
│   │   ├── encoder.py        # PurposeConditionedEncoder, StandardEncoder
│   │   ├── conditioning.py   # FiLM, Concat, Attention conditioners
│   │   ├── task_head.py      # Task prediction heads
│   │   ├── auditor.py        # MultiAttributeAuditor, PostHocAuditorSuite
│   │   ├── baselines.py      # INLP, LEACE baselines
│   │   └── cnn_encoder.py    # CNN encoder for CelebA
│   ├── data/
│   │   ├── base.py           # Base dataset and collate_pcrl_batch
│   │   ├── adult.py          # Adult/Census dataset
│   │   ├── har.py            # HAR (Human Activity Recognition) dataset
│   │   └── celeba.py         # CelebA dataset
│   ├── training/
│   │   ├── trainer.py        # PCRLTrainer, TrainerConfig
│   │   └── losses.py         # Adversarial and task losses
│   ├── evaluation/
│   │   ├── certificates.py   # generate_report(), ComplianceReport
│   │   ├── cvr.py            # Compliance Violation Rate
│   │   ├── probes.py         # Post-hoc linear/MLP probes
│   │   ├── mine.py           # MINE mutual information estimator
│   │   └── visualize.py      # t-SNE, heatmaps, CVR plots
│   ├── purposes/
│   │   ├── spec.py           # PurposeSpec, PurposeRegistry
│   │   ├── composition.py    # Purpose composition operators
│   │   └── verification.py   # Linear/nonlinear compliance certificates
│   └── utils/
│       └── config.py         # PCRLConfig, TrainingConfig, ExperimentConfig
├── configs/                   # YAML experiment configs
├── experiments/               # Experiment scripts
└── tests/                     # Test suite
```

## Quick Start

### 1. Define Purposes

A `PurposeSpec` declares what a representation is for: which tasks it should support, and which sensitive attributes it must not leak.

```python
from pcrl.purposes.spec import PurposeSpec, PurposeRegistry

purpose = PurposeSpec(
    name="income_prediction",
    allowed_tasks=["income"],
    disallowed_attrs=["sex", "race"],
    allowed_task_dims={"income": 2},
    disallowed_attr_dims={"sex": 2, "race": 5},
)

registry = PurposeRegistry()
registry.register(purpose)
```

### 2. Build Models

```python
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.models.auditor import MultiAttributeAuditor

encoder = PurposeConditionedEncoder(
    input_dim=108,
    hidden_dims=[128, 128],
    repr_dim=64,
    num_purposes=1,
    purpose_emb_dim=32,
    conditioning="film",  # or "concat", "attention"
    dropout=0.3,
)

task_heads = {
    "income_prediction": TaskHead(repr_dim=64, output_dim=2),
}

auditors = {
    "income_prediction": MultiAttributeAuditor(
        repr_dim=64,
        attr_output_dims={"sex": 2, "race": 5},
        hidden_dim=256,
        num_layers=3,
    ),
}
```

### 3. Train

```python
from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.training.trainer import PCRLTrainer, TrainerConfig
from torch.utils.data import DataLoader

purposes = get_adult_purposes()
train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
val_ds = AdultDataset(purposes=purposes, root="data", split="val", norm_stats=train_ds.norm_stats)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch)

config = TrainerConfig(
    batch_size=256,
    lr_encoder=1e-3,
    lr_auditor=1e-3,
    lambda_adv=50.0,
    lambda_verify=50.0,
    auditor_steps=10,
    epochs=200,
    early_stopping_patience=15,
    confusion_type="entropy",
)

trainer = PCRLTrainer(
    encoder=encoder,
    task_heads=task_heads,
    auditors=auditors,
    config=config,
    purpose_registry=registry,
    device="cuda",
)

trainer.train(train_loader, val_loader=val_loader)
```

### 4. Evaluate Compliance

```python
from pcrl.evaluation.certificates import generate_report, print_compliance_table

test_ds = AdultDataset(purposes=purposes, root="data", split="test", norm_stats=train_ds.norm_stats)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch)

reports = generate_report(
    encoder=encoder,
    train_loader=train_loader,
    test_loader=test_loader,
    purpose_registry=registry,
    device="cuda",
)
print_compliance_table(reports)
```

Each `ComplianceReport` contains the linear R², empirical best accuracy across post-hoc auditors, and the majority-class baseline. A (purpose, attribute) pair passes compliance when `empirical_best_acc - majority_baseline < 2%` and `linear_r2 < 0.05`.

## Running Experiments

```bash
# Adult dataset (3 purposes: income, employment, education)
python experiments/run_adult.py

# HAR dataset (2 purposes: activity_recognition, health_monitoring)
python experiments/run_har.py

# CelebA dataset
python experiments/run_celeba.py

# Baseline comparisons (Standard, LAFTR, INLP, LEACE)
python experiments/run_baselines.py
```

## Key Concepts

### Purpose Specification

A `PurposeSpec` defines:
- **allowed_tasks**: Tasks the representation should support (e.g., income prediction)
- **disallowed_attrs**: Sensitive attributes that must not be recoverable (e.g., sex, race)
- **allowed_task_dims / disallowed_attr_dims**: Output dimensions for each task/attribute head

### Compliance Audit

The compliance audit (`generate_report`) evaluates each (purpose, attribute) pair using:
1. **Linear R²**: OLS regression of the attribute on the representation — measures linear leakage
2. **Post-hoc auditor suite**: Trains LogisticRegression, RandomForest, SVM (RBF), and XGBoost on the representation to predict the attribute — measures empirical leakage including nonlinear

A pair passes when the best auditor's accuracy is within 2% of the majority-class baseline and the linear R² is below 0.05.

### Conditioning Methods

- **FiLM** (default): Feature-wise Linear Modulation — scales and shifts hidden activations based on purpose
- **Concat**: Concatenates purpose embedding to the input
- **Attention**: Cross-attention between hidden features and purpose embedding

### PCRL vs LAFTR

PCRL uses a `PurposeConditionedEncoder` that produces different representations per purpose. LAFTR uses a `StandardEncoder` with adversarial training to produce a single shared representation that suppresses all disallowed attributes simultaneously.

## Tests

```bash
pytest tests/ -x -q
```

## License

MIT License
