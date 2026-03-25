# PCRL — Claude Code Instructions (Paste These In Order)

Use these prompts one at a time in Claude Code. Wait for each to finish before moving to the next.

---

## Step 1: Project Scaffold

```
Read the IDEA file in this project. Then create a Python research project called "pcrl" with this structure:

pcrl/
├── pcrl/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py          # Purpose-conditioned encoder with FiLM
│   │   ├── task_head.py        # Task prediction heads
│   │   ├── auditor.py          # Adversarial auditors (MLP, RF, XGBoost, SVM)
│   │   └── conditioning.py     # FiLM, concat, and attention conditioning modules
│   ├── purposes/
│   │   ├── __init__.py
│   │   ├── spec.py             # PurposeSpec dataclass and purpose registry
│   │   ├── composition.py      # Compositional purpose algebra (AND, OR, hierarchy)
│   │   └── verification.py     # Compliance certificates and verification
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py        # Synthetic data generator for testing
│   │   ├── adult.py            # Adult/Census dataset loader
│   │   ├── celeba.py           # CelebA dataset loader (placeholder)
│   │   └── base.py             # Base dataset class with purpose specs
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main minimax training loop
│   │   └── losses.py           # Task loss + adversarial loss + verification loss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── cvr.py              # Compliance Violation Rate with diverse auditors
│   │   ├── certificates.py     # Linear and nonlinear compliance certificates
│   │   └── visualize.py        # t-SNE, CVR plots, purpose separation plots
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Configuration dataclasses, YAML loading
├── configs/
│   ├── adult_basic.yaml
│   └── adult_verified.yaml
├── experiments/
│   ├── run_adult.py
│   └── run_synthetic.py
├── tests/
│   └── test_composition.py
├── requirements.txt
└── README.md

Use PyTorch. Type hints everywhere. Make it research-grade code.
The requirements.txt should include: torch, numpy, scikit-learn, xgboost, matplotlib, seaborn, pyyaml, tqdm, pandas.
```

---

## Step 2: Core Abstractions and Purpose Spec

```
Implement pcrl/purposes/spec.py and pcrl/utils/config.py:

1. PurposeSpec dataclass:
   - name: str (e.g., "income_prediction")
   - allowed_tasks: List[str] (e.g., ["income"])
   - disallowed_attrs: List[str] (e.g., ["race", "sex"])
   - task_type: str ("classification" or "regression")
   - allowed_task_dims: Dict[str, int] (output dim per task)
   - disallowed_attr_dims: Dict[str, int] (number of classes per disallowed attr)

2. PurposeRegistry class:
   - register(purpose_spec) — add a purpose
   - get(name) — retrieve by name
   - all_disallowed_for(purpose_name) — return set of disallowed attrs
   - Validate that every purpose has at least one allowed task and one disallowed attr

3. PCRLConfig dataclass:
   - input_dim: int
   - hidden_dims: List[int] (default [256, 256])
   - repr_dim: int (default 128)
   - purpose_emb_dim: int (default 64)
   - conditioning: str ("film", "concat", or "attention")
   - num_purposes: int

4. TrainingConfig dataclass:
   - batch_size: int (default 256)
   - lr_encoder: float (default 1e-3)
   - lr_task: float (default 1e-3)
   - lr_auditor: float (default 1e-3)
   - lambda_adv: float (default 1.0, weight for adversarial loss)
   - auditor_steps: int (default 5, auditor updates per encoder update)
   - epochs: int (default 100)

Include a load_config(yaml_path) function.
```

---

## Step 3: Purpose-Conditioned Encoder

```
Implement pcrl/models/encoder.py and pcrl/models/conditioning.py:

conditioning.py — three interchangeable conditioning modules:

1. FiLMConditioner:
   - Takes purpose embedding e_p and hidden features h
   - Learns gamma(e_p) and beta(e_p) via linear layers
   - Returns gamma * h + beta
   
2. ConcatConditioner:
   - Concatenates purpose embedding with features
   - Projects back to original dim via linear layer

3. AttentionConditioner:
   - Purpose embedding as query, features as keys/values
   - Single-head cross-attention

encoder.py — PurposeConditionedEncoder:
   - Purpose embedding layer: nn.Embedding(num_purposes, purpose_emb_dim)
   - Backbone: MLP with hidden_dims, ReLU activations, BatchNorm
   - Conditioning is injected BETWEEN each hidden layer (not just at input)
   - Forward signature: forward(x, purpose_id) -> h_p
   - The same input x with different purpose_id MUST produce different representations
   
Also implement pcrl/models/task_head.py:
   - TaskHead: small MLP (2 layers) that takes h_p and predicts y_p
   - One TaskHead per (purpose, allowed_task) pair

And pcrl/models/auditor.py:
   - MLPAuditor: neural network auditor for training-time adversarial loss
   - PostHocAuditorSuite: wraps sklearn Random Forest, SVM, XGBoost, Logistic Regression
     - fit(representations, labels) and evaluate(representations, labels) -> dict of metrics
     - Used at evaluation time, NOT during training
```

---

## Step 4: Compositional Purposes (KEY NOVELTY)

```
Implement pcrl/purposes/composition.py:

This is a key novelty of the project. Purposes can compose algebraically.

1. ComposedPurpose class:
   - Supports AND: p1 & p2 means "support tasks from both, hide attrs disallowed by either"
   - Supports OR: p1 | p2 means "support tasks from either, hide only attrs disallowed by both"
   - Supports hierarchy: p_child inherits all constraints from p_parent plus its own
   
   For AND composition:
     allowed_tasks = p1.allowed_tasks UNION p2.allowed_tasks
     disallowed_attrs = p1.disallowed_attrs UNION p2.disallowed_attrs
   
   For OR composition:
     allowed_tasks = p1.allowed_tasks UNION p2.allowed_tasks  
     disallowed_attrs = p1.disallowed_attrs INTERSECTION p2.disallowed_attrs

2. Purpose embedding composition:
   - AND: h_{p1 AND p2} = encoder(x, embed(p1) + embed(p2))  [additive]
   - OR: h_{p1 OR p2} = encoder(x, max(embed(p1), embed(p2)))  [element-wise max]
   - Also try learned composition: embed(p1 AND p2) = MLP([embed(p1); embed(p2)])

3. CompositionLoss:
   - For AND: verify that h_{p1 AND p2} hides everything disallowed by either p1 OR p2
   - For OR: verify that h_{p1 OR p2} hides everything disallowed by both p1 AND p2
   - This is a loss term added during training that enforces compositional consistency

Include a test in tests/test_composition.py that:
   - Creates two purposes with overlapping/non-overlapping disallowed attrs
   - Composes them with AND and OR
   - Verifies the resulting allowed_tasks and disallowed_attrs are correct
```

---

## Step 5: Verification / Compliance Certificates (KEY NOVELTY)

```
Implement pcrl/purposes/verification.py and pcrl/evaluation/certificates.py:

This is the most important novelty. Instead of just claiming "our auditor couldn't break it," we provide mathematical certificates proving disallowed attributes can't be recovered.

verification.py — LinearComplianceCertificate:

1. Given representations H (n x d matrix) and disallowed labels Z:
   - Compute the optimal linear predictor: W* = (H^T H)^{-1} H^T Z
   - Compute the R² score of this predictor
   - If R² < epsilon, issue certificate: "No linear classifier can predict Z from H with R² > epsilon"
   - This is similar to LEACE (Belrose et al. 2023) concept erasure verification

2. NullSpaceCertificate:
   - Project H onto the null space of the linear predictor of Z
   - Measure how much variance is lost
   - Certificate: "The representation's projection orthogonal to Z preserves X% of task-relevant variance"

3. certificates.py — full audit protocol:
   - LinearAudit: run LinearComplianceCertificate for every (purpose, disallowed_attr) pair
   - EmpiricalAudit: run PostHocAuditorSuite (RF, SVM, XGB, LogReg) as sanity check
   - ComplianceReport dataclass:
     - purpose_name: str
     - attr_name: str
     - linear_r2: float
     - linear_certified: bool
     - empirical_best_acc: float
     - empirical_chance_acc: float
     - certified: bool (True if linear_certified AND empirical_best_acc < threshold)
   - generate_report(encoder, dataset, purpose_registry) -> List[ComplianceReport]
   - Print a clean summary table

The key insight: linear certificates are PROVABLE (closed-form), while empirical audits are sanity checks. Together they give both mathematical guarantees and practical confidence.
```

---

## Step 6: Training Loop

```
Implement pcrl/training/losses.py and pcrl/training/trainer.py:

losses.py:
1. TaskLoss: CrossEntropy for classification, MSE for regression
2. AdversarialLoss: CrossEntropy for auditor predicting disallowed attrs
3. CompositionConsistencyLoss: enforces that composed purpose embeddings satisfy the right constraints (from Step 4)
4. VerificationRegularizer (optional): penalizes high linear R² of disallowed attrs during training
   - Compute R² of optimal linear predictor of z from h_p on each batch
   - Add lambda_verify * R² to the encoder loss
   - This directly optimizes toward certifiable compliance

Total encoder loss = TaskLoss - lambda_adv * AdversarialLoss + lambda_comp * CompositionLoss + lambda_verify * VerificationRegularizer

trainer.py — PCRLTrainer:
1. __init__(encoder, task_heads, auditors, purpose_registry, config)
2. train_epoch():
   - For each batch:
     a. For each purpose p:
        - Forward: h_p = encoder(x, p)
        - Compute task loss on allowed tasks
        - For auditor_steps iterations: update auditor to predict disallowed attrs from h_p.detach()
        - Compute adversarial loss (encoder wants auditor to fail)
        - Compute verification regularizer
     b. For each composed purpose pair:
        - Compute composition consistency loss
     c. Update encoder and task heads with combined loss
3. evaluate():
   - Compute task accuracy for each purpose
   - Run full compliance audit (certificates.py)
   - Return metrics dict
4. Log everything with tqdm progress bars
5. Save best model checkpoint based on combined metric (task accuracy * (1 - CVR))
```

---

## Step 7: Synthetic Data for Testing

```
Implement pcrl/data/synthetic.py:

Create a synthetic dataset generator that lets us test everything end-to-end before downloading real data.

SyntheticPCRLDataset:
- Generate n=5000 samples with d=20 features
- Features are grouped:
  - x[0:5] = "task_A features" (correlated with label y_A)
  - x[5:10] = "task_B features" (correlated with label y_B)
  - x[10:15] = "sensitive_1 features" (correlated with sensitive attr z_1)
  - x[15:20] = "sensitive_2 features" (correlated with sensitive attr z_2)
  - Add some cross-correlation so removal is non-trivial

- Two purposes:
  Purpose "task_A": allowed=[y_A], disallowed=[z_1, z_2]
  Purpose "task_B": allowed=[y_B], disallowed=[z_1]

- This means:
  - Under task_A: representation should predict y_A, hide z_1 and z_2
  - Under task_B: representation should predict y_B, hide z_1 but z_2 is fine
  - Under task_A AND task_B: predict both y_A and y_B, hide z_1 and z_2

- Generate labels and sensitive attrs as binary (for simplicity)
- Return as PyTorch Dataset with __getitem__ returning (x, {task_labels}, {sensitive_labels})

Also create experiments/run_synthetic.py:
- Load synthetic data
- Create encoder, task heads, auditors
- Train for 50 epochs
- Run compliance audit
- Print results table showing: for each purpose, task accuracy + CVR + certificate status
- Save t-SNE visualization of representations colored by sensitive attr (should show mixing = good)
```

---

## Step 8: Adult Dataset Integration

```
Implement pcrl/data/adult.py:

Load the Adult Census dataset. For now, if the data files aren't available locally, download them from UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data and adult.test). If the download fails due to network restrictions, create a fallback that generates a synthetic version with the same schema.

AdultPCRLDataset:
- Features: age, workclass, education, education-num, marital-status, occupation, relationship, hours-per-week, capital-gain, capital-loss, native-country
- Encode categoricals with one-hot or ordinal encoding
- Standardize continuous features

- Define three purposes:
  1. "income_prediction": allowed=[income>50K], disallowed=[race, sex]
  2. "employment_analysis": allowed=[occupation_group], disallowed=[race, age_group, marital_status]
  3. "education_assessment": allowed=[education_level], disallowed=[sex, race, income]

- Composition test case: "income_prediction AND employment_analysis"
  - allowed = [income, occupation_group]
  - disallowed = [race, sex, age_group, marital_status]

Create experiments/run_adult.py similar to run_synthetic.py but with real data.
Create configs/adult_basic.yaml and configs/adult_verified.yaml (the verified config enables the verification regularizer).
```

---

## Step 9: Full Evaluation Pipeline

```
Implement pcrl/evaluation/visualize.py and update pcrl/evaluation/cvr.py:

cvr.py — ComplianceViolationRate:
- For a given purpose and trained encoder:
  1. Extract representations for all test data
  2. For each disallowed attribute:
     a. Train 4 auditors: LogisticRegression, RandomForest, SVM-RBF, XGBoost
     b. Each with 3 random seeds = 12 total auditors per attribute
     c. Record best accuracy and AUC across all auditors
  3. CVR = fraction of disallowed attrs where best_accuracy > (chance + margin)
     where margin = 0.05

visualize.py:
1. plot_purpose_separation(encoder, data, purposes):
   - t-SNE of representations, one subplot per purpose
   - Color by sensitive attribute — good representations show NO clustering by sensitive attr
   
2. plot_cvr_comparison(results_dict):
   - Bar chart comparing CVR across methods (PCRL vs baselines)
   
3. plot_certificate_heatmap(compliance_reports):
   - Heatmap: rows = purposes, columns = disallowed attrs
   - Color = linear R² (green = low/certified, red = high/violation)
   - This is the key figure for the paper

4. plot_task_privacy_tradeoff(results_at_different_lambdas):
   - X axis = task accuracy, Y axis = CVR
   - Shows Pareto frontier as lambda_adv varies
```

---

## Step 10: Run Everything End-to-End

```
Run experiments/run_synthetic.py end to end. Fix any bugs. Make sure:
1. Training converges (task loss goes down, auditor loss goes up)
2. Different purposes produce different representations (verify with cosine similarity)
3. CVR is low for disallowed attrs under each purpose
4. Linear certificates pass for disallowed attrs
5. Composition works: AND purpose hides union of disallowed attrs
6. Generate all visualizations and save to results/

Print a final summary table like:
| Purpose | Task Acc | CVR | Linear Cert | Best Auditor Acc (z1) | Best Auditor Acc (z2) |
```

---

## Notes

- If you hit network issues downloading datasets, use synthetic data and move on. Real data comes later.
- The two KEY novelty pieces are: compositional purposes (Step 4) and verifiable compliance certificates (Step 5). Everything else is infrastructure.
- Target: NeurIPS 2025 main track or Trustworthy ML track.
