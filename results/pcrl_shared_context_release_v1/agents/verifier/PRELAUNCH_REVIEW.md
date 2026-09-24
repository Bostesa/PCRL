# Pre-launch verifier review (phase 1)

- **Reviewer:** independent verifier. I did not write any of the code reviewed here.
- **Scope:** the code as committed at `e2f3fba03`. The working tree matched HEAD for `experiments/` and `tests/` when I reviewed it. Written 2026-09-24 ~18:55Z.
- **Access:** read-only for code. No cloud. No inner_check or outer data was read, apart from running the test suite.
- **M3 status:** visible in the code.
  - The closing refit is in `fit_nm._closing_refit` (`fit_nm.py:631-686`). DET_SEL reads the closing bank (`fit_nm.py:755`).
  - The ADV route units are in `selection.py:43` and `runner.py:497-500`.

**Test suite:** `pytest -q tests/pcrl_shared_context_release_v1` gave **106 passed, 1 warning (torch scalar conversion), exit 0** in 116 s.

## Verdict

- **No BLOCKER found.** Role hygiene, legal inputs, the nested law and LP algebra, the paired-oracle and switching rule, contrast signs, screens, ranking, the Bonferroni count and the bootstrap setup all match PROTOCOL sections 6–7 and the design spec.
- **Five MAJOR items.** Two are design asymmetries on the P route that need a decision or an amendment before launch (M-1, M-2). Three are integrity or gating gaps with cheap fixes (M-3, M-4, M-5).

## MAJOR

### M-1. The NM_P and RD_PRIV final checkpoint is chosen by task loss; the P-route ADV is chosen by privacy

- **Where:** `fit_nm.py:689-702` (`select_final_round`). The rule is used for every form, per DESIGN_SPEC §1.
- **What it does:** for `*_P` units the chosen round is the one with the lowest inner_selection task loss among rounds feasible at floor rho−delta. The privacy target (tau on AB/SEX) plays no part in the choice and is not re-checked on the final bank. Every P round sits near the same task cap (D17 + .001), so the rule tends to pick the round with the least AB/SEX protection.
- **The asymmetry:** M3.2 gives ADV_B*_P a privacy-selection rule. NM_P and RD_PRIV keep the task rule.
- **Status:** registered, so this is not a code bug. But it biases the P nominee toward failing its own target, and the two P-route comparators are not treated alike.
- **Decision needed:** amend before launch (for example, a P-form rule of max final-bank AB/SEX slack, then task), or disclose it explicitly.

### M-2. The ADV_B*_P comparators are weakened, which favours NM

This finding comes from the verifier's baseline sub-review.

- **Early stopping always uses the task key**, in both modes (`adv.py:284-285, 321-328`). On the smoke, every run stopped at about epoch 40. The P units also share ADV_B*'s seed (`adv.py:174`), so the privacy rule can only choose among the first ~40 epochs of the same trajectory.
- **Checkpoints are ranked by a moving target.** The shortlist and the final pick use the AB/SEX loss of each epoch's co-trained adversary (`adv.py:336-340, 370-372`). The attackers already refit on each shortlisted law (`adv.py:353-358`) are not used for ranking.
- **Suggested fix:** for `--select privacy`, stop early on the privacy key (or run to the maximum number of epochs), and rank by the refit AB/SEX cut values.

### M-3. Aliases are decided on inner rows only, but the outer assessment reuses them

- **The chain:**
  - `audit_panel.canonicalize` (`audit_panel.py:92-111`) merges releases whose laws are byte-identical on audit_fit, inner_selection and inner_check.
  - `score_outer` then scores only canonical ids (`audit_panel.py:406-408`).
  - `assess` copies the canonical outer scores onto every alias (`assess.py:95-99`).
  - `lock.enumerate_family` merges comparator endpoints through the same aliases (`lock.py:70-72`, from `selection.global_aliases`).
- **Why it matters:** two different functions of x, such as a sparse switched RD or NM law and D17, can agree on every inner person but differ on outer people. The outer endpoint would then score a different function than the locked logical release, and could be labelled `exact_zero_same_route`. MATH_REVIEW §2.5 itself separates a dataset alias from a deployment alias.
- **Likely size:** small (only a handful of outer people would be affected), but the error would be silent.
- **Fix:** in `score_outer`, evaluate every declared logical law on the outer rows. Assert that each is byte-identical to its canonical law, or else refuse to proceed.

### M-4. Positive controls are not required for the lock

This finding comes from the verifier's infra sub-review.

- PROTOCOL §5 and AUDIT_CONTRACT §7 require the six POS cells before the lock.
- The POS units have no dependents (`runner.py:592-595`).
- `lock.build_lock` and `lock.main` (`lock.py:157-201, 254-277`) neither pin nor check POS receipts or `detected` flags.
- The "uninformative for role/anchor" marking is not carried into the lock.
- **Fix:** pin the six POS receipts in the lock, and refuse to lock if any is missing.

### M-5. The outer gate trusts a hand-typed `remote_verified: true`

- **Where:** `audit_panel.py:375-382`.
- **What is missing:** no code checks that `remote_commit_sha` is on origin and contains SELECTION_LOCK.json with the expected SHA.
- **Status:** this follows AR's pattern, so it is procedural only.
- **Fix:** add an `unlock` command that runs `git ls-remote` and hashes `git show <sha>:.../SELECTION_LOCK.json`, then writes the receipt.

## MINOR

1. **Policy tokens are in-sample on the decoder's training rows.** Decoder refits (`fit_nm.py:160-204`) train on nuisance_train under laws whose policy tokens were fit with nuisance_train labels.
   - The decoder can therefore over-trust the policy tokens. That makes policy columns look costlier on coefficient_split, which works against NM.
   - Selection on inner_selection limits the effect. Disclose it.
2. **The decoder records an inner_check task score every round.** `fit_nm.py:615` writes `inner_check_fixed_decoder_task` into ROUND.json. Nothing selects on it (confirmed by grep), but a human sees it before nomination. AR did the same, and AUDIT_CONTRACT §6.3 already discloses it.
3. **The J secondary family can change size.** `assess.py:117-118` drops the J endpoints when `--outer-j` is not given, so the secondary family's size (and its z) differs from the locked list. This family is descriptive only.
4. **Audit seed base is stated wrongly in two docs.** AUDIT_CONTRACT §3 and BASELINE_MATCHING §2 still give 36000; the code and amendment O1 use 26000. Best-response seeds also repeat across anchors (harmless).
5. **`rd.py:357` token-collapse check:** `np.allclose` with rtol 1e-5 can collapse a nearly token-independent attacker, giving an error of about 5e-6 nats, well above PRIMAL_TOL of 1e-7. Use `atol=1e-12, rtol=0`.
6. **ADV privacy fallback is silent.** `adv.py:339, 370` quietly falls back to the task rule. It should be surfaced in selection and the lock.
7. **Watchdog and budget:**
   - The watchdog allows now + 20 h plus 30 min (`cloud.py:187, 215`), which runs past the protocol ceiling of 13:26Z. Clamp it in code.
   - The $50 cap is only recorded.
   - A timeout kills the direct child only (`runner.py:368`).
   - Reusing receipts with `--reuse-commit` hashes only the SC tree, not AR/TAC/TDR (`runner.py:125-128`).
8. **`runner.data_roles` is metadata only.** FIT_ROLES includes inner_check for every fitting unit (`runner.py:456`). Role hygiene is enforced inside the modules, which I checked (see below).

## Verified correct (with evidence)

- **Role hygiene:**
  - inner_check is not read by any fitting or selection step in fit_nm, policies, contexts, rd or adv. Grep finds only the diagnostic at `fit_nm.py:615` and the `rd.py:192` row-fingerprint check.
  - Decoders fit on nuisance_train and are selected on inner_selection. Attackers fit on audit_fit and are validated on inner_selection. The LP, cut floors, DET_SEL and RD_PRIV feasibility use coefficient_split.
  - Rounds and decoders are chosen on inner_selection (`fit_nm.py:580-582, 699`).
  - Outer rows appear only behind `verify_outer_gate` (`audit_panel.py:395, 413-417`). `load_roles` refuses outer labels (`fit_nm.py:121`).
- **Legal inputs:**
  - `policies.check_legal`, `laws.check_inputs` and `NestedLaw.__call__` read only x, ha, T0, p, r and risk.
  - Contexts use x, ha and r through the frozen nuisance.
  - The ADV encoder uses the 49 policy features plus the one-hot D17(T0). H_B reaches only the adversaries.
- **Math:** synthetic check in the scratchpad, `v1.py`.
  - Per-person law, blocks and the direct expected loss agree to 1e-12 for K=4, M=5 with duplicate policy tokens.
  - The LP equalities and signs are right (`channel.py:297-315, 454-472`).
  - U and W normalization is done on the same valid rows (`channel.py:144-153`, `fit_nm.py:274-301`).
  - rho is recomputed from raw blocks on every rebase, including the closing refit.
  - The D17 witness is checked for both eta=0 and eta=1 with the D17 column.
  - The P-form LP returns AB/SEX loss = rho + tau with the task caps held.
  - DET_SEL enumerates all 625 assignments, and its objectives match a person-level replay.
- **Policies:**
  - Paired targets are g − g[D17], and the D17 column is set exactly to 0 (`policies.py:151, 185`).
  - The switch is `gap > tau` with D17 kept on ties (`policies.py:207-216`).
  - g = u − Σλa (`policies.py:287`).
  - In cross-fitting, fold f's decoder is trained on fold ≠ f and scores fold f (`fit_nm.py:330-337`).
  - RD_PRIV prices as Δu − μΔp.
- **Selection, lock and assess:**
  - task = CE_cand − CE_comp and recovery = CE_comp − CE_cand (AR `aggregate_differences`; `lock.py:84-85` plus TAC `contrasts_from_scores`).
  - The U and P screens, H capability ≥ .01 and ranking keys match §6.
  - The DIAGNOSTIC_ONLY fallback, ADV eligibility and rank-minimum rule (M3 route units), and z = norm.isf(.05/(2m)) from the generated list all match.
  - A route passes only if all its clauses pass (TAC `evaluate_family`).
  - The bootstrap is one multinomial over the union of households, 10,000 draws, seed 20260923.
- **Audit panel:**
  - One slate and seed set for all releases. H is fitted once and checked to be identical across releases.
  - The hist_gb leaf rule is a single code path and is asserted against the observed leaf (`audit_panel.py:249-254`).
- **Infra:**
  - SSE on every S3 write.
  - No account-wide destructive calls.
  - Tag-checked terminate.
  - Receipts re-hash inputs, outputs and the code tree on resume.
  - Private paths are git-ignored.
