# Manuscript handoff (atomic; no manuscript or submission metadata was edited)

**Proposed addendum paragraph (negative result, exploratory development on exposed data):**

> Building on the occupation-only adapter, we also adapted both recipients' transmitted predictions: 65 rank-1 income parameters and 325 rank-5 occupation parameters, over frozen trunks and heads. Each recipient had its own anchored NLL, Brier, accuracy and KL ≤ 0.001 budgets. We compared joint, local, task-only, two sequential orders and income-only training in two seeds. The joint candidate kept all 24 original point utility clauses. However, fresh complete-view readers recovered SEX from the aligned income-and-occupation pair only 0.0036 AUC less than from the projected start (simultaneous interval [−0.0025, 0.0097]), an established shortfall against the 0.020 criterion. Joint training did not beat local or sequential training. Changing income gave no coalition reduction, and the candidate resolved fewer occupation utility clauses than the start (3/12 vs 10/12). The design was chosen after the earlier results, and the intervals are conditional on the fitted models and readers.

**Insert location:** suggested for the supplementary negative-results section, next to the occupation-only adapter. This is the author's decision; this campaign made no edit.

**Claims that must not be made:**
- a population privacy guarantee;
- novelty of the mechanism;
- that income caused or did not cause the earlier shortfall;
- that coalition signal is irremovable in general;
- calling S12/S21 Taylor et al.'s algorithm.
