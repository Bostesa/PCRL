# PENDING_INTEGRATION — Study 6 (utility-first residual extension `[H_A, Z_J, R]`)

State when v5 was published (2026-09-19, ~18:00Z): Terminal 1 branch `research/pcrl-utility-extension-aws-v1`
is at `7f961d5c` (identical to the Study 5 handoff). Its handoff reports tier 0: inventory done, **AWS
blocked (CLI session expired; the user must run `aws login`)**. No protocol, gate file, fit or outcome
exists. Nothing from Study 6 is in the v5 PDF except the prospective criteria of §9.

## Where to look (read-only; fetch small reports only)
* Shared status: `<git-common-dir>/pcrl_parallel_handoff_v5/terminal_1/STATUS.json` and `HISTORY.jsonl`
  (fields `milestone`, `evidence_commit`, `tier`).
* Evidence: `git show <evidence_commit>:results/<study dir>/<file>`; never copy the archive. Expected
  small files: PROTOCOL/METHOD, pilot gate JSON, COUNTS, decision document, interval CSVs, 2017 panel if
  triggered. If a file lives only in private S3, fetch that single object by hash into the scratchpad, verify
  sha256, analyse, delete the local copy, and record it in `MANUSCRIPT_STORAGE_LEDGER.md`.

## Checks before filling anything
1. Protocol and gate committed before the first pilot outcome (compare commit time to first fit record).
2. Gate file machine-readable; inputs = reconstruction, source eligibility, validation sensitive increments;
   **no residence/commute field read**.
3. Deployed D0: fixed function of permitted A-side inputs; no out-of-fold target, row id, `H_B` or sensitive
   label at inference; replay test present.
4. Both increments per endpoint: over H and over J. Old-view attacks routed as candidates onto the new view.
5. Utility audit includes the old-view predictor as a candidate (or reports both).
6. Counts cumulative (pilot slots inside the 135-slot plan counted once); untriggered tiers labelled
   `NOT TRIGGERED`.
7. Utility-first criterion: one-sided adjusted upper bounds on every declared sensitive endpoint; .001
   allowance named; all strong external comparators and local controls; ordinary (non-coalition) extension
   compared.
8. 2016 untouched.

## Commands
```
cd <manuscript worktree>
git fetch origin && cat <git-common-dir>/pcrl_parallel_handoff_v5/terminal_1/STATUS.json
# write results/pcrl_manuscript_review_v6/checks/verify_study6.py modelled on v5's verify_study5.py
#   (SHA = evidence_commit; read via `git show`; assert every printed number)
python results/pcrl_manuscript_review_v6/checks/verify_study6.py
cd papers/pcrl_manuscript_v6 && latexmk -pdf main.tex
```
Use one CPU worker and one BLAS thread. No model fitting.

## Which §9/Conclusion text replaces the neutral one (choose exactly ONE after evidence)
Kept out of the PDF until the evidence chooses.

**(a) Negative pilot (gate not passed; expansion NOT TRIGGERED).** "The utility-first pilot did not pass its
preregistered validation gate on [criterion]; the expanded family was not run and is reported as not triggered,
not as missing evidence. No utility-first claim is made."

**(b) Failed expanded study.** "The gate passed and the expanded family ran; no prospectively selected
extension showed a utility gain over J with added sensitive recovery bounded by one-sided adjusted intervals on
every declared endpoint [give the binding endpoint and bound]."

**(c) Competitive development candidate.** "Extension [config], chosen without residence or commute labels,
adds [gain] nats of residence capability over J [interval], with added recovery over J bounded above by
[bounds] on all declared endpoints (the .001-nat allowance is an allowance, not zero harm), against every strong
external comparator and local control. Over H its total recovery is [Δ_H]. It is a development candidate on
repeatedly used pools; a fresh prospective evaluation is required, and 2016 remains sealed."

**(d) Utility gain without coalition specificity.** As (c), followed by: "An ordinary (non-coalition) residual
extension performs [equally/within interval]; the gain is therefore not attributable to coalition
conditioning, and no coalition-specific claim is made."
