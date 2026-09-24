# Baseline/audit smoke notes (engineering only, not science)

Mac, 2026-09-24 18:00–18:35Z. Anchor 0, 30% household subsample (`fit_nm.load_roles(smoke=True)`, identical to NM smoke rows), OMP_NUM_THREADS = 2. Rows touched: nuisance_train, audit_fit, coefficient_split and inner_selection for fitting and selection. inner_check was touched only by the positive control score and by one law-dispatch shape check (no loss computed). Outer was never touched. Outputs are in the session scratchpad only (`scratchpad/baseline_smoke/private/`), never in the repo. Banks used:

- a reduced AR-layout smoke bank (`rd --variant SMOKE_BANK`, sources H and D17 only; 20.7 s);
- the method implementer's M2 bank `scratchpad/smoke/bank_a0_m2` (`fit_nm build-bank --smoke`). `rd.load_bank` read it through `fit_nm.load_bank_dir` (inventory verified), and `check_bank_roles` confirmed identical role fingerprints.

## Structural findings that changed the competitors (all before any science fit)

1. **RD_TASK, pre-M2 oracle.** tau = 0.002 moved 82–89% of people off D17, with inner receiver task no better than D17. Added the registered D17-shrinkage grid tau ∈ {0.002, 0.01, 0.03, 0.1, 0.3}. After M2 (paired oracle) the NM d_1 moves 10.4% of people.
2. **RD_PRIV.** Frozen-attacker prices point away from D17: round 0 was feasible at mu = 0 on the frozen bank, and after one best response it needed mu = 77.6 (95% of people changed) and still failed its own refit by +0.008 nats. Added the registered tau-bisection fallback plus the D17 alias fallback. With the fallback, round 1 kept 3.8–7.7% of people changed and passed the final bank, which includes its own closing best response.
3. **RD scoring.** Coverage-rule decoders understate re-assigned policies, so RD candidates are now scored by pure-law receivers. Per-candidate receiver seeds gave exact-D17 candidates up to −0.0023 nats of spurious advantage, so one common receiver seed is used per unit.
4. **ADV.**
   - At lr 1e-3 and weight decay 1e-4, the encoder and decoder overfit: at beta = 0, training task fell 0.43 → 0.39 while inner_selection task rose 0.475 → 0.502 in 15 epochs. Changed to lr 3e-4 and weight decay 1e-3.
   - The encoder could not reproduce D17 without T0: 47% of its warm-start mass was off D17. Added onehot(D17(T0)), a legal input; the warm-start mass off D17 is now 3%.
   - The adversary warm start left adversaries worse than the prior on inner_selection (AB/SEX CE 0.90–1.14). Warm starts are now early-stopped, and AB/SEX sits at 0.683–0.70.
   - Under the spec checkpoint rule (lowest task), every smoke run picked the warm start: task 0.4639–0.4643, with the D17 receiver at 0.4636. Adversarial training raised own-adversary AB/SEX CE from 0.683 to 0.702 at +0.013 task over 40 epochs, so ADV's P-route shot needs the optional `--select privacy` rule.
5. **Positive controls (smoke).** AB/SEX: +0.692 / +0.691 nats (U / PWGTP). AB/RAC1P: +1.215 / +1.194. Both were detected.
6. **Laws dispatch.** `laws.write_descriptor` + `laws.person_law` worked for `deterministic_policy` (RD) and `adv_mlp` (ADV). Row-sum error was ≤ 2.2e-16. RD support is 1. ADV has maximum support 17 and mean 16.1 (see the hist_gb caveat). `hb` is refused with PermissionError.
7. **Pickling.** `python -m ...rd` pickled policies under `__main__`, and the laws dispatcher could not load them. Fixed by dispatching `main()` through the package module.

## Timing (smoke, 2 threads, wall / user CPU) and 1-CPU full-data estimates

Full data has about 3.3× the rows. Slate fits scale roughly linearly. Estimates allow ×1.5 for one AWS thread vs the Mac. Peak RSS was 0.36–0.45 GB per process.

| Unit | Smoke wall / CPU | Estimate per unit, full data, 1 CPU |
|---|---|---|
| RD_TASK (4 rounds × 5 tau receivers + 6 cross-fit slates + D17 receiver; smoke ran 2 rounds) | 30 s / 49 s | 6–10 min |
| RD_PRIV (setup + 4 rounds × (bisection + 4 best-response slates + 1 receiver); smoke ran 2 rounds) | 45 s / 74 s | 8–12 min |
| ADV per epoch | 0.32–0.42 s | 1.1–2 s → ≤ 300 epochs ≤ 10 min; the patience-40 stop typically comes at 40–80 epochs (≈ 2–3 min) |
| ADV warm starts | 10 s | ≈ 0.5–1 min |
| ADV best responses (per shortlisted checkpoint, 4 slates on a support-16 law) | 21–29 s | 1.5–2.5 min × 4 = 6–10 min |
| ADV unit total | 75–104 s (≤ 40 epochs, 2 shortlisted) | 10–20 min |
| Positive control per cell (5 slates, deterministic label law) | 3–9 s | ≈ 0.5–1 min (6 cells ≈ 5 min) |

Per anchor: 2 RD + 2 ADV + 2 positive controls ≈ 35–65 CPU-minutes. Three anchors ≈ 2–3.3 CPU-hours, and every unit is single-threaded and independent.
