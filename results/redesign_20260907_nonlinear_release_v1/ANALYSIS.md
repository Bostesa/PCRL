# Nonlinear protection release: held-out analysis

Completed seeds: 0, 1, 2. All utility/leakage values are predictive R² on the final test split. Means use paired seed observations; SD is sample SD, undefined for a single seed. Negative R² is retained.

Threshold decisions use unrounded scores; all table values and SDs are displayed to six decimal places. For E_selected, seed 1 P1→U test R²=0.9899977563666246; seed 2 P1→U test R²=0.9899693218116854. These values are below 0.99 and fail the fresh-test task floor; rounding must not turn them into passes.

The validation utility floor is **0.99 for each purpose separately**. Protection requires every P1→V,S and P2→U,S score ≤0.05 and combined→S ≤0.10 for **both linear and MLP probes**. The combined recipient may access U,V. These thresholds define the empirical pilot criteria; they are not universal privacy guarantees.

## All predeclared controls, references and exploratory configurations

Every fixed/dual configuration is shown. Test performance does not choose the primary configurations. Saved prior references retain their original training history; matched comparisons are identified separately. Oracle and prediction-only controls are reference releases with different information/dimension constraints, not equivalent learned representation methods.

| Method | Release dimensions P1/P2/combined | P1→U MLP task | P2→V MLP task | Worst linear leakage | Worst MLP leakage | Combined S linear | Combined S MLP |
|---|---|---:|---:|---:|---:|---:|---:|
| A_oracle | 1/1/2 | 0.999990 ± 0.000002 | 0.999994 ± 0.000004 | 0.000055 ± 0.000291 | -0.000131 ± 0.000476 | -0.001506 ± 0.001678 | -0.003137 ± 0.000688 |
| B_prediction_only | 1/1/2 | 0.997930 ± 0.000097 | 0.996819 ± 0.000643 | 0.000076 ± 0.000345 | -0.000304 ± 0.000561 | -0.001437 ± 0.001501 | -0.001906 ± 0.001522 |
| C_saved_task_only | 8/8/16 | 0.997570 ± 0.000230 | 0.996245 ± 0.000820 | 0.004199 ± 0.007697 | 0.350252 ± 0.062275 | -0.000236 ± 0.008356 | 0.156921 ± 0.061658 |
| R_prior_0.1 | 8/8/16 | 0.914579 ± 0.109144 | 0.972785 ± 0.013152 | -0.000647 ± 0.002591 | 0.273950 ± 0.124572 | -0.007234 ± 0.003106 | 0.115385 ± 0.033735 |
| R_prior_1 | 8/8/16 | 0.917500 ± 0.097650 | 0.906028 ± 0.050135 | 0.000078 ± 0.003707 | 0.252139 ± 0.201772 | -0.007392 ± 0.005192 | 0.097947 ± 0.005746 |
| C_matched_task_only | 8/8/16 | 0.990291 ± 0.007611 | 0.990237 ± 0.004300 | 0.002965 ± 0.004073 | 0.331666 ± 0.024183 | -0.003709 ± 0.003739 | 0.147321 ± 0.106316 |
| D_fixed_0.01 | 8/8/16 | 0.993397 ± 0.001826 | 0.988222 ± 0.006908 | 0.004832 ± 0.004587 | 0.246872 ± 0.041821 | 0.000195 ± 0.004481 | 0.182220 ± 0.120963 |
| D_fixed_0.1 | 8/8/16 | 0.990725 ± 0.002055 | 0.989049 ± 0.003158 | 0.005315 ± 0.005170 | 0.184891 ± 0.048654 | -0.001517 ± 0.001348 | 0.058150 ± 0.015039 |
| E_dual_0.01 | 8/8/16 | 0.991629 ± 0.001444 | 0.989006 ± 0.005376 | 0.003616 ± 0.005801 | 0.219025 ± 0.014146 | 0.000528 ± 0.004725 | 0.178686 ± 0.083030 |
| E_dual_0.1 | 8/8/16 | 0.991721 ± 0.001500 | 0.988302 ± 0.003741 | 0.005259 ± 0.005316 | 0.166506 ± 0.039396 | -0.001822 ± 0.001582 | 0.058804 ± 0.009669 |

## Previously validation-selected configurations

A selection is eligible as a primary result only if both purpose utility floors passed on validation. If no candidate met that floor, the saved fallback is diagnostic only. All such selected observations remain visible below; their test outcomes are not used to replace the selection.

| Selected family | Validation utility eligible | Validation protection pass | Validation jointly feasible | P1→U MLP task | P2→V MLP task | Worst linear leakage | Worst MLP leakage | Combined S linear | Combined S MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_selected | 2/3 | 0/3 | 0/3 | 0.991865 ± 0.003484 | 0.992330 ± 0.000208 | 0.005099 ± 0.005048 | 0.230895 ± 0.064414 | -0.002564 ± 0.000775 | 0.102306 ± 0.080658 |
| E_selected | 2/3 | 0/3 | 0/3 | 0.990760 ± 0.001345 | 0.992254 ± 0.000265 | 0.003941 ± 0.006339 | 0.203319 ± 0.034479 | -0.002589 ± 0.001003 | 0.107760 ± 0.043096 |

| Seed | Family | Saved configuration | Selection status |
|---|---|---|---|
| 0 | D_selected | D_fixed_0.01 | primary_utility_eligible |
| 1 | D_selected | D_fixed_0.1 | diagnostic_only_utility_floor_failed |
| 2 | D_selected | D_fixed_0.01 | primary_utility_eligible |
| 0 | E_selected | E_dual_0.01 | primary_utility_eligible |
| 1 | E_selected | E_dual_0.1 | diagnostic_only_utility_floor_failed |
| 2 | E_selected | E_dual_0.01 | primary_utility_eligible |

## Utility and protection criteria by arm

Counts below are validation decisions. Passing leakage while failing either task floor does not satisfy the intended joint criterion.

| Arm | Utility floor passed | Protection passed | Jointly feasible | Per-seed utility/protection/joint |
|---|---:|---:|---:|---|
| A_oracle | 3/3 | 3/3 | 3/3 | 0: 1/1/1; 1: 1/1/1; 2: 1/1/1 |
| B_prediction_only | 3/3 | 3/3 | 3/3 | 0: 1/1/1; 1: 1/1/1; 2: 1/1/1 |
| C_saved_task_only | 3/3 | 0/3 | 0/3 | 0: 1/0/0; 1: 1/0/0; 2: 1/0/0 |
| R_prior_0.1 | 0/3 | 0/3 | 0/3 | 0: 0/0/0; 1: 0/0/0; 2: 0/0/0 |
| R_prior_1 | 0/3 | 0/3 | 0/3 | 0: 0/0/0; 1: 0/0/0; 2: 0/0/0 |
| C_matched_task_only | 1/3 | 0/3 | 0/3 | 0: 0/0/0; 1: 1/0/0; 2: 0/0/0 |
| D_fixed_0.01 | 2/3 | 0/3 | 0/3 | 0: 1/0/0; 1: 0/0/0; 2: 1/0/0 |
| D_fixed_0.1 | 0/3 | 0/3 | 0/3 | 0: 0/0/0; 1: 0/0/0; 2: 0/0/0 |
| E_dual_0.01 | 2/3 | 0/3 | 0/3 | 0: 1/0/0; 1: 0/0/0; 2: 1/0/0 |
| E_dual_0.1 | 0/3 | 0/3 | 0/3 | 0: 0/0/0; 1: 0/0/0; 2: 0/0/0 |
| D_selected | 2/3 | 0/3 | 0/3 | 0: 1/0/0; 1: 0/0/0; 2: 1/0/0 |
| E_selected | 2/3 | 0/3 | 0/3 | 0: 1/0/0; 1: 0/0/0; 2: 1/0/0 |

## Paired comparisons

Changes are left minus right within seed. Positive utility change favors the left arm; negative leakage change lowers that probe's predictive success. Sample SD is computed on paired changes. These are all declared paired observations, including any that failed the utility floor; eligible-pair counts prevent treating diagnostic fallbacks as successful primary results.

| Comparison | Utility-eligible pairs | P1→U MLP task | P2→V MLP task | Worst linear leakage | Worst MLP leakage | Combined S linear | Combined S MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
| E_dual_0.01_minus_D_fixed_0.01 | 2/3 | -0.001769 ± 0.001332 | +0.000783 ± 0.001533 | -0.001216 ± 0.001749 | -0.027848 ± 0.029769 | +0.000333 ± 0.000246 | -0.003534 ± 0.059012 |
| E_dual_0.1_minus_D_fixed_0.1 | 0/3 | +0.000996 ± 0.000600 | -0.000746 ± 0.000634 | -0.000056 ± 0.000189 | -0.018385 ± 0.018488 | -0.000305 ± 0.000300 | +0.000654 ± 0.006215 |
| E_selected_minus_D_selected | 2/3 | -0.001105 ± 0.002408 | -0.000076 ± 0.000075 | -0.001158 ± 0.001805 | -0.027575 ± 0.030165 | -0.000025 ± 0.000374 | +0.005454 ± 0.056628 |
| D_selected_minus_C_matched_task_only | 0/3 | +0.001575 ± 0.007911 | +0.002093 ± 0.004142 | +0.002135 ± 0.001108 | -0.100771 ± 0.064062 | +0.001145 ± 0.003528 | -0.045016 ± 0.140178 |
| E_selected_minus_C_matched_task_only | 0/3 | +0.000469 ± 0.006872 | +0.002017 ± 0.004140 | +0.000976 ± 0.002844 | -0.128347 ± 0.036174 | +0.001120 ± 0.003832 | -0.039562 ± 0.141425 |

## Native direct prediction versus independently fitted task probes

Native U/V predictive R² is reported separately from the MLP that is trained to recover U/V from the release. An oracle receives the true permitted signal; prediction-only uses the saved task prediction. Neither control is evidence for a richer representation's novelty.

| Control | Direct P1→U test R² | Task-probe P1→U MLP R² | Direct P2→V test R² | Task-probe P2→V MLP R² |
|---|---:|---:|---:|---:|
| A_oracle | 1.000000 ± 0.000000 | 0.999990 ± 0.000002 | 1.000000 ± 0.000000 | 0.999994 ± 0.000004 |
| B_prediction_only | 0.997769 ± 0.000124 | 0.997930 ± 0.000097 | 0.996770 ± 0.000689 | 0.996819 ± 0.000643 |

## Attacker competence positive control

The exposed-target diagnostic gives each view U,V,S directly and uses the same independent attacker budgets. Recovery here checks basic probe competence; good recovery does not prove those probes detect every nonlinear code. Oracle/prediction-only leakage remains reported in the main table.

| Split | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |
|---|---|---:|---:|---:|---:|---:|
| validation | linear | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| validation | mlp | 0.999308 ± 0.000137 | 0.999151 ± 0.000275 | 0.999047 ± 0.000078 | 0.999199 ± 0.000115 | 0.999399 ± 0.000191 |
| test | linear | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| test | mlp | 0.999303 ± 0.000136 | 0.999167 ± 0.000304 | 0.999090 ± 0.000076 | 0.999228 ± 0.000071 | 0.999408 ± 0.000157 |

## Training adversary learning and transfer

Before warmup, after warmup, and final values are maxima of the normalized 1−MSE training surrogate over the five prohibited targets on a fixed training batch. They are not held-out predictive scores. Final is measured after the last training-eraser refresh. Transfer and independent-audit columns are predictive R² on the same validation rows and final calibrated release: the first reuses the frozen training adversaries and their training normalization, while the second uses newly fitted validation-selected MLP auditors. The final training task pair is also an in-sample diagnostic, separate from task-probe utility.

A weak training adversary or poor transfer across the final eraser can hide leakage from the optimization objective. The independent audit remains the selection/evaluation evidence; neither a low training surrogate nor a low transfer score establishes protection.

| Seed | Arm | Initial surrogate max | After warmup max | Final surrogate max | Transfer validation max | Independent MLP validation max | Final training task U/V | Final λ min/max |
|---|---|---:|---:|---:|---:|---:|---|---|
| 0 | C_matched_task_only | -0.043591 | 0.131834 | 0.319648 | 0.276549 | 0.333669 | 0.965394/0.964985 | 0.000000/0.000000 |
| 0 | D_fixed_0.01 | -0.043591 | 0.131834 | 0.268413 | 0.171556 | 0.229427 | 0.953457/0.912304 | 0.010000/0.010000 |
| 0 | D_fixed_0.1 | -0.043591 | 0.131834 | 0.106722 | 0.048470 | 0.178881 | 0.847602/0.654249 | 0.100000/0.100000 |
| 0 | E_dual_0.01 | -0.043591 | 0.131834 | 0.275186 | 0.167983 | 0.239882 | 0.912163/0.918312 | 0.008660/0.020396 |
| 0 | E_dual_0.1 | -0.043591 | 0.131834 | 0.052786 | 0.046842 | 0.168651 | 0.875938/0.622701 | 0.071906/0.120004 |
| 1 | C_matched_task_only | -0.016880 | 0.097515 | 0.438697 | 0.266577 | 0.332567 | 0.825848/0.984457 | 0.000000/0.000000 |
| 1 | D_fixed_0.01 | -0.016880 | 0.097515 | 0.225792 | 0.137477 | 0.277342 | 0.726860/0.938367 | 0.010000/0.010000 |
| 1 | D_fixed_0.1 | -0.016880 | 0.097515 | 0.079686 | 0.014854 | 0.149160 | 0.835799/0.991197 | 0.100000/0.100000 |
| 1 | E_dual_0.01 | -0.016880 | 0.097515 | 0.195972 | 0.141906 | 0.269483 | 0.717712/0.951025 | 0.011850/0.017982 |
| 1 | E_dual_0.1 | -0.016880 | 0.097515 | 0.069760 | 0.076048 | 0.158552 | 0.852750/0.990841 | 0.076083/0.101426 |
| 2 | C_matched_task_only | 0.109557 | 0.175020 | 0.360091 | 0.229378 | 0.324848 | 0.831211/0.719228 | 0.000000/0.000000 |
| 2 | D_fixed_0.01 | 0.109557 | 0.175020 | 0.196367 | 0.143285 | 0.284374 | 0.885523/0.849666 | 0.010000/0.010000 |
| 2 | D_fixed_0.1 | 0.109557 | 0.175020 | 0.150813 | 0.014007 | 0.247275 | 0.971866/0.860743 | 0.100000/0.100000 |
| 2 | E_dual_0.01 | 0.109557 | 0.175020 | 0.182845 | 0.047024 | 0.252487 | 0.890449/0.858150 | 0.014248/0.027133 |
| 2 | E_dual_0.1 | 0.109557 | 0.175020 | 0.141337 | 0.004976 | 0.218687 | 0.974120/0.880556 | 0.081859/0.160959 |

## Each prohibited target

| Arm | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |
|---|---|---:|---:|---:|---:|---:|
| A_oracle | linear | 0.000002 ± 0.000361 | -0.000836 ± 0.000834 | -0.001490 ± 0.000981 | -0.001727 ± 0.001855 | -0.001506 ± 0.001678 |
| A_oracle | mlp | -0.000457 ± 0.000784 | -0.003288 ± 0.001408 | -0.002083 ± 0.001672 | -0.003583 ± 0.003822 | -0.003137 ± 0.000688 |
| B_prediction_only | linear | 0.000029 ± 0.000405 | -0.000784 ± 0.000736 | -0.001419 ± 0.000951 | -0.001711 ± 0.001773 | -0.001437 ± 0.001501 |
| B_prediction_only | mlp | -0.000505 ± 0.000798 | -0.003394 ± 0.001181 | -0.002055 ± 0.001369 | -0.003727 ± 0.003647 | -0.001906 ± 0.001522 |
| C_saved_task_only | linear | 0.002766 ± 0.008974 | -0.002058 ± 0.002010 | -0.004613 ± 0.010162 | -0.003663 ± 0.000953 | -0.000236 ± 0.008356 |
| C_saved_task_only | mlp | 0.319128 ± 0.078422 | 0.094268 ± 0.069239 | 0.318713 ± 0.058038 | 0.102933 ± 0.058169 | 0.156921 ± 0.061658 |
| R_prior_0.1 | linear | -0.000817 ± 0.002829 | -0.003848 ± 0.000423 | -0.002302 ± 0.003968 | -0.004824 ± 0.001750 | -0.007234 ± 0.003106 |
| R_prior_0.1 | mlp | 0.197327 ± 0.200118 | 0.099257 ± 0.050326 | 0.141856 ± 0.083518 | 0.109417 ± 0.063163 | 0.115385 ± 0.033735 |
| R_prior_1 | linear | 0.000078 ± 0.003707 | -0.003589 ± 0.000676 | -0.007028 ± 0.004485 | -0.006508 ± 0.003016 | -0.007392 ± 0.005192 |
| R_prior_1 | mlp | 0.217581 ± 0.234509 | 0.067559 ± 0.030100 | 0.088288 ± 0.081597 | 0.084369 ± 0.016206 | 0.097947 ± 0.005746 |
| C_matched_task_only | linear | 0.002520 ± 0.004473 | -0.002516 ± 0.002998 | -0.006762 ± 0.008489 | -0.002260 ± 0.003741 | -0.003709 ± 0.003739 |
| C_matched_task_only | mlp | 0.325276 ± 0.032319 | 0.066199 ± 0.034708 | 0.263657 ± 0.062827 | 0.083314 ± 0.064481 | 0.147321 ± 0.106316 |
| D_fixed_0.01 | linear | 0.003743 ± 0.005679 | -0.002121 ± 0.002469 | -0.005101 ± 0.010535 | -0.001038 ± 0.000667 | 0.000195 ± 0.004481 |
| D_fixed_0.01 | mlp | 0.246872 ± 0.041821 | 0.058588 ± 0.016918 | 0.202655 ± 0.040308 | 0.066278 ± 0.041665 | 0.182220 ± 0.120963 |
| D_fixed_0.1 | linear | 0.004243 ± 0.005956 | -0.000950 ± 0.002843 | -0.004746 ± 0.011319 | -0.003878 ± 0.001295 | -0.001517 ± 0.001348 |
| D_fixed_0.1 | mlp | 0.135888 ± 0.037628 | 0.021993 ± 0.010152 | 0.147705 ± 0.085498 | 0.029884 ± 0.015528 | 0.058150 ± 0.015039 |
| E_dual_0.01 | linear | 0.003102 ± 0.006053 | -0.001827 ± 0.002039 | -0.005364 ± 0.009860 | -0.001434 ± 0.000882 | 0.000528 ± 0.004725 |
| E_dual_0.01 | mlp | 0.214114 ± 0.006029 | 0.043400 ± 0.012916 | 0.180315 ± 0.047516 | 0.058562 ± 0.026176 | 0.178686 ± 0.083030 |
| E_dual_0.1 | linear | 0.004139 ± 0.006073 | -0.001126 ± 0.002576 | -0.004467 ± 0.010252 | -0.004286 ± 0.001051 | -0.001822 ± 0.001582 |
| E_dual_0.1 | mlp | 0.130795 ± 0.034115 | 0.030312 ± 0.008944 | 0.126600 ± 0.080733 | 0.030837 ± 0.011461 | 0.058804 ± 0.009669 |
| D_selected | linear | 0.004010 ± 0.006130 | -0.003644 ± 0.001352 | -0.005244 ± 0.010447 | -0.002137 ± 0.002377 | -0.002564 ± 0.000775 |
| D_selected | mlp | 0.230895 ± 0.064414 | 0.048867 ± 0.021184 | 0.181511 ± 0.059123 | 0.050721 ± 0.049487 | 0.102306 ± 0.080658 |
| E_selected | linear | 0.003427 ± 0.006609 | -0.003192 ± 0.001508 | -0.005451 ± 0.009813 | -0.002275 ± 0.002218 | -0.002589 ± 0.001003 |
| E_selected | mlp | 0.198408 ± 0.028172 | 0.035880 ± 0.001321 | 0.171412 ± 0.056028 | 0.048213 ± 0.029454 | 0.107760 ± 0.043096 |

## Per-seed, per-purpose results

| Seed | Arm | P1→U MLP task | P2→V MLP task | Worst linear leakage | Worst MLP leakage | Combined S linear | Combined S MLP | Validation P1→U | Validation P2→V | Validation utility/protection/joint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | A_oracle | 0.999992 | 0.999998 | -0.000181 | -0.000547 | -0.000540 | -0.003931 | 0.999992 | 0.999998 | 1/1/1 |
| 1 | A_oracle | 0.999987 | 0.999993 | 0.000379 | 0.000388 | -0.003444 | -0.002754 | 0.999990 | 0.999994 | 1/1/1 |
| 2 | A_oracle | 0.999991 | 0.999990 | -0.000034 | -0.000235 | -0.000536 | -0.002724 | 0.999993 | 0.999987 | 1/1/1 |
| 0 | B_prediction_only | 0.997819 | 0.996556 | -0.000197 | -0.000621 | -0.000622 | -0.000149 | 0.997815 | 0.996034 | 1/1/1 |
| 1 | B_prediction_only | 0.997965 | 0.997551 | 0.000464 | 0.000344 | -0.003169 | -0.002797 | 0.997873 | 0.997673 | 1/1/1 |
| 2 | B_prediction_only | 0.998004 | 0.996349 | -0.000039 | -0.000636 | -0.000518 | -0.002771 | 0.998128 | 0.996751 | 1/1/1 |
| 0 | C_saved_task_only | 0.997569 | 0.995460 | 0.000173 | 0.408853 | -0.007984 | 0.099951 | 0.997714 | 0.994745 | 1/0/0 |
| 1 | C_saved_task_only | 0.997341 | 0.997096 | 0.013074 | 0.357045 | 0.008618 | 0.222387 | 0.997239 | 0.997018 | 1/0/0 |
| 2 | C_saved_task_only | 0.997801 | 0.996180 | -0.000650 | 0.284859 | -0.001343 | 0.148423 | 0.997896 | 0.996564 | 1/0/0 |
| 0 | R_prior_0.1 | 0.992783 | 0.957818 | -0.003011 | 0.413876 | -0.005291 | 0.081854 | 0.992544 | 0.955940 | 0/0/0 |
| 1 | R_prior_0.1 | 0.961067 | 0.978042 | 0.002122 | 0.232859 | -0.005595 | 0.114982 | 0.960851 | 0.980076 | 0/0/0 |
| 2 | R_prior_0.1 | 0.789888 | 0.982495 | -0.001052 | 0.175115 | -0.010817 | 0.149321 | 0.784147 | 0.982748 | 0/0/0 |
| 0 | R_prior_1 | 0.994934 | 0.903152 | -0.003231 | 0.479575 | -0.004003 | 0.094339 | 0.994248 | 0.894361 | 0/0/0 |
| 1 | R_prior_1 | 0.949766 | 0.957539 | 0.004084 | 0.182202 | -0.004803 | 0.094928 | 0.950798 | 0.962086 | 0/0/0 |
| 2 | R_prior_1 | 0.807800 | 0.857393 | -0.000619 | 0.094640 | -0.013369 | 0.104573 | 0.802471 | 0.867815 | 0/0/0 |
| 0 | C_matched_task_only | 0.995592 | 0.990423 | 0.000848 | 0.359244 | -0.007939 | 0.048704 | 0.995445 | 0.988872 | 0/0/0 |
| 1 | C_matched_task_only | 0.993711 | 0.994440 | 0.007660 | 0.314078 | -0.000846 | 0.259947 | 0.993350 | 0.993362 | 1/0/0 |
| 2 | C_matched_task_only | 0.981570 | 0.985846 | 0.000387 | 0.321678 | -0.002342 | 0.133313 | 0.981987 | 0.986861 | 0/0/0 |
| 0 | D_fixed_0.01 | 0.995385 | 0.992184 | 0.001890 | 0.234305 | -0.003053 | 0.055192 | 0.995340 | 0.992108 | 1/0/0 |
| 1 | D_fixed_0.01 | 0.993013 | 0.980246 | 0.010117 | 0.212776 | 0.005307 | 0.296031 | 0.992778 | 0.983324 | 0/0/0 |
| 2 | D_fixed_0.01 | 0.991795 | 0.992238 | 0.002490 | 0.293536 | -0.001671 | 0.195439 | 0.991809 | 0.992147 | 1/0/0 |
| 0 | D_fixed_0.1 | 0.991398 | 0.988117 | 0.000730 | 0.149464 | -0.000302 | 0.044129 | 0.991290 | 0.988148 | 0/0/0 |
| 1 | D_fixed_0.1 | 0.988417 | 0.992568 | 0.010918 | 0.164844 | -0.002968 | 0.056287 | 0.987845 | 0.991707 | 0/0/0 |
| 2 | D_fixed_0.1 | 0.992359 | 0.986461 | 0.004296 | 0.240367 | -0.001281 | 0.074033 | 0.991712 | 0.987151 | 0/0/0 |
| 0 | E_dual_0.01 | 0.992313 | 0.992022 | -0.001344 | 0.208278 | -0.002866 | 0.117882 | 0.992690 | 0.991912 | 1/0/0 |
| 1 | E_dual_0.01 | 0.992604 | 0.982799 | 0.009996 | 0.213745 | 0.005925 | 0.273284 | 0.992519 | 0.985307 | 0/0/0 |
| 2 | E_dual_0.01 | 0.989969 | 0.992197 | 0.002197 | 0.235051 | -0.001476 | 0.144893 | 0.990155 | 0.992077 | 1/0/0 |
| 0 | E_dual_0.1 | 0.992424 | 0.986900 | 0.000456 | 0.127048 | -0.000261 | 0.048398 | 0.992052 | 0.986998 | 0/0/0 |
| 1 | E_dual_0.1 | 0.989998 | 0.992542 | 0.010970 | 0.166629 | -0.003424 | 0.060504 | 0.989284 | 0.991607 | 0/0/0 |
| 2 | E_dual_0.1 | 0.992740 | 0.985465 | 0.004350 | 0.205841 | -0.001781 | 0.067511 | 0.992066 | 0.987238 | 0/0/0 |
| 0 | D_selected | 0.995385 | 0.992184 | 0.001890 | 0.234305 | -0.003053 | 0.055192 | 0.995340 | 0.992108 | 1/0/0 |
| 1 | D_selected | 0.988417 | 0.992568 | 0.010918 | 0.164844 | -0.002968 | 0.056287 | 0.987845 | 0.991707 | 0/0/0 |
| 2 | D_selected | 0.991795 | 0.992238 | 0.002490 | 0.293536 | -0.001671 | 0.195439 | 0.991809 | 0.992147 | 1/0/0 |
| 0 | E_selected | 0.992313 | 0.992022 | -0.001344 | 0.208278 | -0.002866 | 0.117882 | 0.992690 | 0.991912 | 1/0/0 |
| 1 | E_selected | 0.989998 | 0.992542 | 0.010970 | 0.166629 | -0.003424 | 0.060504 | 0.989284 | 0.991607 | 0/0/0 |
| 2 | E_selected | 0.989969 | 0.992197 | 0.002197 | 0.235051 | -0.001476 | 0.144893 | 0.990155 | 0.992077 | 1/0/0 |

## Per-seed native direct predictions

| Seed | Control | Validation P1→U | Validation P2→V | Test P1→U | Test P2→V |
|---|---|---:|---:|---:|---:|
| 0 | A_oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | A_oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | A_oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0 | B_prediction_only | 0.997703 | 0.996001 | 0.997671 | 0.996455 |
| 1 | B_prediction_only | 0.997646 | 0.997496 | 0.997727 | 0.997560 |
| 2 | B_prediction_only | 0.998080 | 0.996573 | 0.997908 | 0.996294 |

All values, paired per-seed differences, saved selection records, source provenance and input metric hashes are available in release_summary.json. This report performs no fitting or selection and preserves the original result files.
