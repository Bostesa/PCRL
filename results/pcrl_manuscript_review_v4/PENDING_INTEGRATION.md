# PENDING_INTEGRATION — Study 5 (Terminal 1, research/pcrl-competitive-method-v1)

State at writing (2026-09-18): branch at `4f09e63c6` contains only a verified selection diagnosis of
Study 4 (CORRECTIONS.md, SELECTION_DIAGNOSIS.md, CHECKPOINT_LEDGER.json). **No METHOD, PROTOCOL, MATRIX or
PROTOCOL_FREEZE; no fit; no outcome.** Nominal counts (168 neural, 204 projection/control/diagnostic slots)
are a plan and are not treated as evidence of execution.

## Before filling any cell of Table 7 (tables/study5_pending.tex)
1. Protocol + freeze hash committed BEFORE the first fit; check freeze timestamp < first fit record.
2. Locked counts: planned / released / distinct / distinct-and-new, from identity records (not score agreement).
3. Prospective reductions only, each with measured-runtime justification recorded before outcomes.
4. Handoff items B1–B5 (`.git/pcrl_parallel_handoff_v4/terminal_2/HANDOFF.json`) resolved or disclosed:
   one declared correction level with both reported; measured rank/support/tolerance; projection form
   declared (subtractive vs whitened); mass-scaled alias registered as alias; correction family intercept.
5. Track N: J weights hash-verified; common frozen A0 teacher; γ=0 keeps source losses; heads read Z only;
   step 0 equal budget; attacker machinery unchanged.
6. Primary candidate selected without residence/commute labels or evaluation scores; one config across seeds.
7. Primary scope kernel_expanded_independent; unchanged A0/J reproduce their matched audit exactly.
8. 2016 untouched.

## Integration commands
```
EV=<T1 results dir>
python results/pcrl_manuscript_review_v4/checks/verify_study4.py $EV   # adapt: same checks, new family names
python results/pcrl_manuscript_review_v4/checks/build_assets.py        # extend with Study 5 tables/figures
cd papers/pcrl_manuscript_v4 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Then choose exactly ONE of the two conclusion paragraphs in main.tex §13 ("How this conclusion will change")
according to whether a candidate meets the four-level criteria in §10.
