# Saved-adversary inspection: seed 2, E_dual_0.01

This inspection reproduces saved predictions without fitting a model or eraser, changing historical files, or accessing final-test examples. Exact values, source hashes, preprocessing arrays, and all sensitivity scores are in `inspection.json`.

Both A (saved training adversary) and D (saved independent MLP audit) receive the **same final calibrated validation release** and 2,048 examples. They use raw-target predictive R² with the same evaluation-mean-centered denominator. No target-column or scorer discrepancy was found.

| Prohibited target | A validation R² | D validation R² |
|---|---:|---:|
| p1_V | 0.013991081362404 | 0.210284278117756 |
| p1_S | 0.027885659348826 | 0.040498195299549 |
| p2_U | 0.047024004933239 | 0.252486578007161 |
| p2_S | 0.014323280898098 | 0.080119768074708 |
| combined_S | 0.012826009952113 | 0.128558142534542 |

A reproduced exactly; D reproduced within 8.88e-16. The highlighted P2→U MSEs are 0.9069139242090274 (A) and 0.7113823794595707 (D), with common validation variance 0.9516650250413635.

## Training and audit differences

| Factor | Saved training adversary A | Independent audit D |
|---|---|---|
| Targets and heads | P1: V/S; P2: U/S; combined: S (2/2/1 outputs) | U/V/S for every view (3/3/3 outputs) |
| Architecture | 32×32 ReLU; P2 has 1,410 parameters | 32×32 ReLU; P2 has 1,443 parameters |
| Fitting examples | 3,072 representation-training rows, indices 1024–4095 | 2,048 independent attacker-fitting rows |
| Release distribution | Co-adapting encoder; LEACE/statistics refresh every 25 encoder updates | Frozen final encoder and separately calibrated final LEACE |
| Target objective | Mean of five standardized-target MSEs across three networks | Per-view mean of three standardized-target MSEs |
| Adam | lr=.001, default betas/eps, no decay; one optimizer for three networks | Same Adam hyperparameters; separate optimizer per view/start |
| Updates and exposure | 900: 100 warmup + 800 interleaved; 230,400 row presentations, each update-pool row exactly 75 times | 640 per start × 2 starts = 1,280 per view; 327,680 row presentations |
| Selection | One initialization; fixed final state; no validation selection | Per-target minimum validation MSE over two starts and 17 epochs/start |
| P2→U selected state | Final moving-training state | Seed 700260, epoch 80, optimizer step 640 |
| Feature coordinates | Training-eraser holdout, standard-deviation floor 1e-6, refreshed through step 400 | Attacker-fitting final releases, fixed statistics and inactive-feature tolerance 1e-10 |
| Target coordinates | Fixed mean/std from representation-training update pool | Mean/std from attacker-fitting rows |

The saved training target U mean/std are 0.010814960114657879 / 1.011916995048523; the audit U mean/std are 0.009497739774301506 / 0.994125941840819. Both predictors correctly invert their own scaling before scoring. The local training surrogate `1−normalized MSE` is distinct from the published A validation transfer score.

## Refresh and map sensitivity

The last 50 adversary updates used map/statistics 375. Encoder update 400 is followed by refresh 400, **after adversary update 900**, with no subsequent adversary optimization. Final calibration then supplies another map.

| Map applied to the fixed final encoder | Feature statistics | P2→U validation R² |
|---|---:|---:|
| final_calibration | 375 | 0.052381447514147 |
| final_calibration | 400 | 0.047024004933239 |
| training_375 | 375 | 0.192595684834663 |
| training_375 | 400 | 0.176400078693749 |
| training_400 | 375 | 0.097052028655489 |
| training_400 | 400 | 0.069500774928784 |

Changing only final-release normalization from step 400 to step 375 moves P2→U R² from .047024 to .052381. The larger changes with old maps also change released information and map-to-encoder alignment. They are sensitivity diagnostics, not a causal decomposition, and do not reconstruct the unsaved encoder399/adversary training inputs.

## Loading recipe

Use the historical method directory `results/redesign_20260907_nonlinear_release_v1/seed_2/E_dual_0.01`. Load tensors with `torch.load(..., map_location="cpu", weights_only=True)`. Original imported source hashes match the saved frozen hash manifest.

```python
checkpoint = torch.load(source / "training/final.pt", map_location="cpu", weights_only=True)
encoder = PerPurposeLoRAEncoder(SharedEncoder(), 2, rank=4, alpha=4., dropout=0.)
encoder.load_state_dict(checkpoint["encoder"]); encoder.eval()
adversaries = ReleaseAdversaries(8, 32)
adversaries.load_state_dict(checkpoint["adversaries"]); adversaries.eval()
data, manifest = run_nonlinear_release.make_data(2)  # include_test=False
raw = [run_nonlinear_conflict.encode(encoder, data["validation"]["x"], p) for p in range(2)]
# Load source/erasers.npz. For each p: final_h=(raw_h-center) @ matrix.T + center.
# Saved A: normalize final_h with checkpoint["training_release"] feature_mean/std;
# concatenate normalized views; predict; inverse-scale using target_mean/std.
# Output mapping: network0=(V,S), network1=(U,S), network2=(S).
# D: load each probes/attack_{p1,p2,combined}_mlp/preprocessing.npz into
# FitPreprocessing; load selected_target_{0,1,2}.pt state_dicts into MLPProbe.
# Each selected checkpoint outputs all three targets: U,V,S; take that target's
# own column, then inverse-scale with the probe's own saved target statistics.
# Score raw targets by 1-MSE/validation_target_variance, without clipping.
```

## Review of the new diagnostic runner

No blocking fitting, sealed-test, or source-mutation issue was found. Continuations receive only cached attacker-fitting and validation releases. Saved maps are never fitted again. B/C clone source models and reset Adam; original coordinates are preserved. D reloads the exact saved target checkpoints. New diagnostic test generation follows saved attacker selections and any predeclared bridge decision. Source tensors, buffers, maps and representative validation outputs are checked unchanged.

Interpretation limits remain: B/C match added optimizer effort, while B retains 900 prior updates; D has different target sharing and coordinates; and the optional bridge changes both feature and target normalization, including relative raw-target loss weights. Success after extra fitting on fixed releases cannot alone establish that moving representations caused the original gap.
