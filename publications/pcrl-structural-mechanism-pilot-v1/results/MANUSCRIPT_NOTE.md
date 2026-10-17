# Note for the manuscript owner (secondary task; nothing applied to the manuscript)

This campaign did not edit the revised manuscript (`papers/combined_empirical_v2/` on `research/combined-paper-advisor-revision-v1`, GitHub `9e6d01f`) and adds none of its exploratory results to it.

## 1. Delta-review item closed: the public-code sentence

- **Manuscript text** (Reproducibility paragraph, `main.tex` line 700 at `9e6d01f`): "The encoders' public code still exposes an earlier bound from linear $R^2$ to classifier accuracy that was later shown not to hold; we do not use it."
- **Re-check.** On 2026-10-17 at about 01:45 UTC, `git ls-remote origin refs/heads/main` gave `55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827` on GitHub. At that commit:
  - `pcrl/purposes/verification.py` line 208 defines `certified_accuracy_bound`, headed "Theorem: Linear Compliance Guarantee";
  - `pcrl/evaluation/certificates.py` imports it;
  - the retirement patch (`results/pcrl_claims_foundation_v1/patches/0001-retire-accuracy-guarantee-on-main.patch` on `research/pcrl-claims-foundation-v1`) has not been applied, and `docs/ACCURACY_CERTIFICATE_RETIREMENT.md` does not exist on main.
- **Status: the sentence is accurate as of that commit.** Re-check it again before submission if main changes.

## 2. Delta-review item still open

The claim-to-source rows corrected after the claims reviewer's second pass, and rows 47–51 of `results/combined_paper_advisor_revision_v1/CLAIM_SOURCE_CHECK.md`, have still not been re-reviewed. This campaign did not take that review on.

## 3. A precise motivation paragraph (suggested text, within the paper's evidence cutoff)

It uses no ACS result, because the paper's evidence ends with the useful-head comparison. It fits the Introduction after the paragraph that states the scoped promises.

> A recipient that is permitted to use a release for one purpose may still be able to learn something it is not permitted to learn. Our recipients hold exactly what they are sent: features, a model output, or both, computed from fields that are statistically associated with a declared protected attribute. A task that the protected attribute helps to predict will produce useful outputs that vary with features that also predict that attribute, so a recipient with a labelled sample of people who received the same release can learn to read the attribute from what it is legitimately given. Two recipients that serve the same people can read more from their linked releases than either can alone. We study this access because it is what recipients actually hold, plus a labelled auxiliary sample and, for pairs, linkage, which we grant explicitly. We do not study it because it yields the strongest attack: recipients with other side information, model parameters or query access face different threat models that we do not evaluate. Our measurements are what the declared readers recover from the declared release, reported beside the release's usefulness.
