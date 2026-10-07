# Advisor brief — lra (Adult learned decoder and constrained release)

**Bottom line.** The real Adult test ran end to end, and no method claim passed. This closes this exact recipe. It is
not a general negative for learned decoders or constrained search.
- **Claims.** Claim A was not established, on precision. Claims B and C had no eligible nominee.
- **Decoder.** Learning the probabilities did not clearly improve confidence. On this development assessment, the
  learned decoder (κ = 32, fitted on the rows the frozen heads were trained on) made held-out confidence worse on every
  named same-token map (nominal 95%).
- **Nominated release.** P\* is the existing cbp JOINT λ0.1 map with its original mean decoder. Its partition was
  privacy-trained with SEX, and it reads no true task label.
  - It was nominated by a selection rule chosen after cbp, which is adaptive.
  - On the registered slate it cuts pair attribute recovery by 0.034 AUC (registered lower bound 0.029) below the
    strongest task-only compression. That matches qpc's earlier result on the same rows.
  - It misses the full criterion only because its occupation log-loss preservation is unresolved: upper bound 0.0121
    against the 0.01 limit.
- **Label.** CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION. Q (the fixed original task-only code) passed,
  repeating earlier results on the same reused rows.

## What was tested

- **The question.** Can a learned decoder (certified, class-preserving, convex for each fixed token), plus assignment
  search under explicit confidence and information budgets, give a more private release that keeps both tasks'
  predictions and confidence?
- **The comparison bank.** It was complete:
  - old maps;
  - the same maps with the learned decoder;
  - a supervised task-only compression;
  - 72 weighted controls;
  - 15 constrained fits (local, both sequential orders, joint single-move, joint paired-move);
  - the usual references.
- **The launch gate.** It was correctness-only: twelve implementation checks on the four known-law fixtures, all passed
  and independently reproduced. It did not demand a synthetic win. The predecessor's registered trigger had made that
  unattainable by construction on its fixture bank (PREDECESSOR_GATE_DIAGNOSIS.md).

## What happened

1. **The learned decoder lost held-out confidence.** This is consistent with overfitting, and applies to this decoder,
   these maps and this assessment.
   - On the fitting rows it cut occupation log loss by about 0.026 nats on identical tokens. Those are the rows the
     frozen teacher heads were trained on.
   - On held-out rows it added log loss: +0.010 nats on the inner-named fixed-map control (D1 JOINT λ0.01) and +0.024
     on the supervised task map.
   - Every named contrast's nominal 95% interval excludes zero (supplementary; outside the 37-slot family).
   - Information is unchanged, because the tokens are identical; this is purely a confidence loss.
2. **The supervised assignment search also lost held-out confidence.**
   - The new task-only map is worse held out than the old privacy-untrained FINE-TASK map, even with the old decoder:
     occupation excess 0.015 against 0.003.
   - The constrained arms met their hard fitting budgets in-sample on every seed. Held out, their occupation log-loss
     excess was 0.048–0.061 nats (point estimates across the five arms). The two scored fallbacks are
     MEASURED_VIOLATION against 0.01.
3. **No method component met its registered criterion.**
   - Neither constrained nor weighted search produced an inner-eligible release, so claim B had no eligible nominee.
     That is not a head-to-head loss.
   - Paired joint search was descriptively indistinguishable from single-move and sequential search (pair AUC 0.792 vs
     0.791–0.796). Claim C was not tested, because J\* had no eligible nominee.
4. **The engineering held up.**
   - Independent replay re-certified every decoder unit (81 fixed-map + 90 new).
   - It replayed 185,057 search states in 42 of the 90 mapping units, including all 99 paired moves. That covers all
     constrained and C-TASK units, plus the weighted controls at λ 0.025 and 0.08.
   - It also reproduced every selection value and the evaluation lock, and all real-data controls passed.
   - Independent reproduction of the assessment endpoints (verifier phase 3): VERIFIER_PHASE3_STATUS.

## Scope (prompt §17)

- A decoder-only utility change is not information removal.
- A fixture result is not an Adult result.
- An Adult development result is not confirmation.
- A useful calibrated existing code is not a newly invented algorithm. The nominated P\* is an existing map.
- Fewer leaking finite attackers is not a population privacy guarantee.
- This is not "calibration can never help". This decoder was fitted on rows the teacher had already seen.

## Recommendation

- **Close this recipe.** No further λ or κ tweak on this assessment.
- **If calibration is revisited:** it needs held-out calibration data the heads never saw. That is a new, separately
  registered design.
- **No confirmation population is spent.**

## Cost and custody

- About 6.8 CPU-h of 20 at the inference point (final figure in COST_AND_CLOSEOUT.md); at most two processes; $0 cloud.
- Custody: a verified same-device copy, restored from (teacher, learned decoders, P\*, attacker); off-device backup
  pending because the drive was absent (BACKUP_VERIFICATION.json).
- No novelty is claimed. Privacy Funnel (arXiv:1402.1774), Taylor–Vippathalla–Coon (arXiv:2601.21859v2), proper-loss
  calibration and greedy partition search are established ideas (PRIOR_ART_AND_CLAIM_SCOPE.md).
