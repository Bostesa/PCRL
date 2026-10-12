# Useful service and original utility

The admitted U0 uses source encoders plus fitted StandardScaler/LogisticRegression heads. Affine folding uses W=A/s and b=c−Wm. Binary logits are (−d/2,+d/2), multiclass logits centered in pinned class order. The restored source parity and independent role checks precede new fitting. Encoder arithmetic is float32 and heads float64.

Income classes are <=50K/ >50K; occupation has six fixed categories. Accuracy uses first argmax, NLL floors probabilities at 1e−12, and multiclass Brier sums across classes. Neither source nor new releases need true labels at inference. Source models saw all FIT rows; internal selection is not independent of source pretraining. Historical head selection and global calibration exposure are disclosed in EXPOSURE_AND_ROLE_MAP.md.

U_NEW and U_TASKPOWER selected exact epoch-zero aliases; their trained trajectories are preserved. Every primary nominee passes all 24 point clauses on INNER against BOTH original U0 and frozen Ucal. Confidence interval feasibility remains separate. JOINT has 13 PASS/11 UNRESOLVED in K51, and 7 PASS/17 UNRESOLVED under all-arm K240. Zero unresolved clauses are relabeled as violations. Per-seed/task metrics, balanced accuracy, class recalls, probability confidence and constant diagnostics are reported in the R5 closeout tables.

The FIT fold-two criterion is only a common U0-relative training selection surrogate. It is not an original-reference FIT conjunction. No INNER-strength replacement or best-seed selection occurred. All fallbacks were frozen before task labels; none was activated. Historical H_RAW_CSTAR passed 16/24 and failed 8/24 and remains descriptive.

Engineering amendment H preserves original reference probability bytes under distinct reference keys, preventing a U0-candidate save from overwriting reference keys. V1 outputs are preserved; V2 scalar utility/inference outputs are authoritative after independent replay. No model/reader, nomination, cap, endpoint family or scientific selection changed.
