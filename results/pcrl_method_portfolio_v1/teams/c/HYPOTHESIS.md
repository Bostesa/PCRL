# Team C: purpose-specific grouped feature selection

Status: Phase I design draft, 2026-10-08. No Adult fit or screen result is claimed here.

## Question

Can two fixed, purpose-specific subsets of the permitted Adult input features support useful, calibrated income and occupation predictions while disclosing less SEX information to each recipient and their coalition than full-feature, utility-only, independent, sequential, and simple common-removal controls?

The information-changing operation is deletion of entire raw feature groups before either task head is trained or run. A mask is fixed for all people within a fitted release, never chosen using a person's protected label. SEX may be used to train the selector on the allowed defense-fit role; it is not a deployment input.

## Pre-fit premise

The admitted predictor schema has 83 encoded columns from five numeric and five categorical source fields. `education` and `education-num` express the same education fact, so Team C binds them into one selection atom. There are nine toggleable atoms. `relationship` has a near-direct SEX proxy in the admitted Adult data (`Husband` and `Wife` categories); both candidate masks exclude it by design, before results. This premise is historical development knowledge, not fresh evidence. If this exclusion costs too much utility, the candidate fails rather than receiving a changed threshold.

Each recipient receives its selected encoded raw values and a head output computed only from those values. For deterministic heads and public fixed masks, the coalition's ideal information about SEX is exactly that in the union of selected raw values: head outputs are deterministic functions of values already in the coalition view. A finite attacker may still benefit from the head outputs, so measured attacks must include them. A purpose-specific mask can lower a local view without lowering coalition leakage if another recipient retains the group.

## Falsifiable forecast

Dropping the `relationship` atom from both masks should reduce raw-feature SEX recovery. It may also breach the tight occupation or income confidence limits. Optional grouped removals selected jointly should be most useful where independently chosen masks retain complementary proxy groups in their union. The matched fixed `relationship`-removal arm may explain the whole gain; in that case report a simple feature-removal result, not a joint-design advance.

The candidate earns an exploratory method lead only under the common protocol's utility clauses against both U0 and Ucal and a fresh complete-view privacy gain over the strongest applicable eligible feature-track control. Joint attribution additionally requires a measured advantage over both sequential orders, the independent selector, and the common-removal selector. No fairness, mutual-information, or privacy guarantee follows from feature deletion alone.

## Kill rules

- No useful predictions and confidence under the unchanged common mean-utility caps.
- Fresh nonlinear complete-view attacks recover SEX as well as the strongest control, including from the raw union or either recipient.
- A common fixed group removal, utility-only mask, independent mask, or sequential order accounts for the gain.
- The selected mask accidentally includes a label, SEX, or an incomplete categorical/education fact group.

No Team C design is a confirmation study. Adult outcomes and source designs have already been used repeatedly.
