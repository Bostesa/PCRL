# PCRL: Purpose-Conditioned Representation Learning

A PyTorch framework for learning fair representations that are conditioned on data processing purposes, enabling purpose-specific information filtering.

## Overview

PCRL implements a purpose-conditioned encoder that learns representations optimized for specific data processing purposes while protecting sensitive attributes. The framework uses adversarial training to ensure that disallowed sensitive information cannot be recovered from the learned representations.

### Key Features

- **Purpose-Conditioned Encoding**: Representations are conditioned on the specified purpose using FiLM, concatenation, or attention mechanisms
- **Adversarial Training**: Auditors attempt to predict sensitive attributes, encouraging the encoder to remove this information
- **Multiple Purposes**: Support for multiple purposes with different allowed tasks and disallowed attributes
- **Compliance Violation Rate (CVR)**: Metric to measure how well sensitive attributes are protected
- **Post-hoc Probes**: Linear and MLP probes for evaluating representation quality

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
pcrl/
├── pcrl/
│   ├── models/
│   │   ├── encoder.py      # Purpose-conditioned encoder
│   │   ├── task_head.py    # Task prediction heads
│   │   └── auditor.py      # Adversarial auditors
│   ├── data/
│   │   ├── base.py         # Base dataset class
│   │   ├── adult.py        # Adult/Census dataset
│   │   └── celeba.py       # CelebA dataset
│   ├── training/
│   │   ├── trainer.py      # Main training loop
│   │   └── losses.py       # Loss functions
│   ├── evaluation/
│   │   ├── cvr.py          # Compliance Violation Rate
│   │   └── probes.py       # Post-hoc probes
│   └── utils/
│       └── config.py       # Configuration dataclasses
├── configs/
│   └── adult_example.yaml  # Example configuration
├── experiments/
│   └── run_adult.py        # Adult dataset experiment
└── tests/
```

## Quick Start

### 1. Define Purpose Specifications

```python
from pcrl import PurposeSpec

purpose = PurposeSpec(
    name="income_prediction",
    allowed_tasks=["income"],
    disallowed_attrs=["sex", "race"],
    task_type="classification",
)
```

### 2. Configure the Model

```python
from pcrl import PCRLConfig, TrainingConfig, ExperimentConfig

model_config = PCRLConfig(
    input_dim=108,
    hidden_dims=[256, 128],
    repr_dim=64,
    purpose_emb_dim=32,
    purposes=[purpose],
    conditioning="film",  # or "concat", "attention"
)

training_config = TrainingConfig(
    batch_size=256,
    lr_encoder=1e-3,
    lr_auditor=1e-3,
    lambda_adv=1.0,
    auditor_steps=5,
    epochs=100,
)

config = ExperimentConfig(
    model=model_config,
    training=training_config,
)
```

### 3. Load Data and Train

```python
from pcrl import AdultDataset, create_trainer

train_dataset = AdultDataset(
    purposes=[purpose],
    root="data",
    split="train",
)

trainer = create_trainer(
    config=config,
    train_dataset=train_dataset,
    device="cuda",
)

trainer.train()
```

### 4. Evaluate

```python
from pcrl import evaluate_all_purposes

cvr_results = evaluate_all_purposes(
    encoder=trainer.encoder,
    config=model_config,
    test_dataset=test_dataset,
)

print(f"CVR: {cvr_results['income_prediction'].overall_cvr:.4f}")
```

## Running Experiments

### Adult Dataset

```bash
python experiments/run_adult.py --config configs/adult_example.yaml
```

With custom parameters:

```bash
python experiments/run_adult.py \
    --epochs 50 \
    --batch-size 128 \
    --lambda-adv 2.0 \
    --device cuda
```

## Configuration

Configurations can be specified via YAML files:

```yaml
model:
  input_dim: 108
  hidden_dims: [256, 128]
  repr_dim: 64
  purpose_emb_dim: 32
  conditioning: film
  purposes:
    - name: income_prediction
      allowed_tasks: [income]
      disallowed_attrs: [sex, race]
      task_type: classification

training:
  batch_size: 256
  lr_encoder: 0.001
  lr_auditor: 0.001
  lambda_adv: 1.0
  auditor_steps: 5
  epochs: 100
```

## Key Concepts

### Purpose Specification

A `PurposeSpec` defines:
- **allowed_tasks**: Tasks that can be predicted from the representation
- **disallowed_attrs**: Sensitive attributes that should NOT be predictable
- **task_type**: Classification or regression

### Compliance Violation Rate (CVR)

CVR measures how well sensitive attributes are protected:

```
CVR = (auditor_accuracy - random_baseline) / (1 - random_baseline)
```

- CVR = 0: Perfect protection (auditor no better than random)
- CVR = 1: No protection (auditor achieves perfect accuracy)

### Conditioning Methods

- **FiLM**: Feature-wise Linear Modulation (default)
- **Concat**: Simple concatenation of purpose embedding
- **Attention**: Cross-attention based conditioning

## License

MIT License
