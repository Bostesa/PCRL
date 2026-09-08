# Application-screen protocol and chronology

This directory reproduces earlier exploratory oracle-label checks. Their protocol
description is RETROSPECTIVE, not a predeclared method comparison. Earlier values
were already viewed before this file was written. They are retained in metrics.json
and checked against the previous observations rather than silently replaced.

The additional learned-output reference is fixed before execution: only official
HAR training subjects, seed9138 permuted into13 task-fit/4 attacker-fit/4 development
subjects; no official test files. Fit one StandardScaler+binary LogisticRegression
(C1,lbfgs,tol1e-4,max_iter2000) on task-fit subjects to predict active/sedentary.
Freeze it; attack activity from hard output or20 fixed equal-width probability
bins using add-one conditional counts fitted on the attacker subjects. Evaluate
on the last4 subjects. No model, threshold, bin-count, or task selection on these
outcomes. These attacks are bounded diagnostics, not universal guarantees.

Oracle checks use only HAR official train labels and Diabetes processed train
labels, RNG9137 sequentially,70/30 fitting/development row partitions, add-one
conditional-count attackers. These random HAR windows do not establish session
generalization. The learned reference uses different, disjoint participant groups.

The application decision concerns missing evidence for reusable-release necessity;
it does not prove rich releases are unnecessary. No PCRL or other representation
method is run. No real-data application pilot is selected. No final-test results
are used. Inventory counts/names include existing test files; only explicitly
listed training file contents are read or hashed. Configuration, source hash,
data hashes, coverage, exact scores, and compute time are saved. Historical files
are never modified. No downloads or dependencies are installed.
