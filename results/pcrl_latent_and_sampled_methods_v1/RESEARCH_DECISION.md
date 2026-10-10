# Research decision: latent repair and sampled confidence

**Status:** Exploratory development result on repeatedly used Adult rows. No assessment or unexposed population was opened. The old feature-mask and reference-repair findings remain historical evidence, not new outcomes of this campaign.

## Feature-code comparison

We completed two paired-seed real-data fits and fresh complete-view attacks for task-only T, fixed-kernel D-K, local L, both freeze-first sequential orders S12/S21, ordinary adversarial C, and unchanged continuous F. The six hard-code arms disclose at most 1,024 code states per recipient, plus probabilities and decisions derived solely from their codes. F discloses 16 scaled continuous latent values per recipient plus its task output. R04 independently replayed all seven releases, attacks and the full fixed 567-clause simultaneous family.

| Arm | Pair SEX AUC, seed 0 / seed 1 | Utility status |
| --- | ---: | --- |
| T, task-only ten-bit | 0.832582 / 0.835960 | Fails full point gate |
| D-K, kernel repair ten-bit | 0.805905 / 0.793778 | Occupation confidence violation |
| L, local ten-bit | 0.817431 / 0.796456 | Fails full point gate |
| S12 / S21, sequential ten-bit | 0.802288 / 0.797881; 0.811717 / 0.797484 | Both fail full point gate |
| C, adversarial ten-bit | 0.743901 / 0.764677 | Fails full point gate |
| F, continuous latent | 0.869335 / 0.862196 | All 24 point clauses pass; 9/36 interval clauses unresolved |

The strongest favorable *descriptive* matched-capacity contrast is T minus D-K pair AUC: **0.026676 / 0.042181** by seed, mean **0.034429 [0.013719, 0.055138]**. The interval crosses the required 0.020 benefit. D-K's mean occupation NLL excess over U0 is **0.032253 [0.010368, 0.054137]** against a 0.010 cap, and its mean occupation Brier excess over Ucal is **0.014797 [0.006994, 0.022600]** against a 0.005 cap. These lower bounds establish utility violations. All six ten-bit arms fail the full point utility gate, leaving no eligible hard-code matched control or candidate. C has lower pair AUC than D-K in both seeds, with worse utility; the kernel and joint components earned no competitive advantage.

F is the strongest favorable utility example, but its wider wire and higher pair AUC make it a different service. F's 24/24 point utility clauses pass; the simultaneous interval conjunction is not established because nine clauses cross their limits. A descriptive cross-wire D-K versus Ucal score anchor reduction is **0.056925 [0.034748, 0.079102]** in pair AUC, but D-K fails utility and the contracts differ. It is not a competitive claim.

## Sampled score channel

The locked finite score channel changes the transmitted confidence value by sampling a public output-bank code. It is score-only; it does not protect feature sharing. The 16 bounded fits covered joint stochastic, task-only stochastic, joint deterministic, task-only deterministic, product, local and both freeze-first sequential arms on two seeds. Fourteen completed searches selected no mean-utility-feasible policy. Both S21 jobs ended after two full infeasible stage-one starts; their stage two did not run, so their full fits are technically incomplete. No protected sampled policy, sampled release, protected-release attack or mechanism-variance estimate exists from those fits. This is a bounded search result, not a proof that all stochastic channels fail.

The fixed historical score references and identity/class-only diagnostics have real ATT and INNER releases, 18 fresh attack cells and independent full-view/reader/AUC replay. The task-only T* pair AUC is **0.846745/0.829728**; inherited P* is **0.802702/0.803579** and meets the point utility caps on both tasks and both references in both seeds. Class-only is **0.736103/0.734125** but fails task utility. Identity preserves useful point task metrics, but it leaves the fine confidence information. The independent endpoint replay matched all 1,161 estimable registered clause bounds, with 5,724 fitted-arm clauses unavailable. These fixed cells cannot demonstrate a randomness or joint-design advantage for a policy that was not selected.

## Claims and next step

- **Competitive method:** None established in these two tested recipes.
- **Privacy-training component:** D-K lowers measured recovery relative to T at a confirmed occupation utility cost; no eligible benefit.
- **Joint design:** No gain established over eligible matched local and both sequential controls; the hard-code eligible set is empty.
- **Randomness:** No claim. No fitted protected sampled release reached deployment or fresh attack.
- **Uncertainty:** Intervals are conditional on fitted models/readers and reused development people; they do not absorb adaptive search, training variation or population transport.

**One next step:** design a utility-feasible representation or output bank using an explicit prefit utility witness, then freeze one comparison against task-only, local and both sequential controls on development rows. Any eventual confirmation requires separate authorization for a genuinely unexposed population and a prospective frozen protocol.
