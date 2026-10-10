# R3 conditional protection and matched controls

Status: prefit design for the current exploratory Adult development study. R3 may run data-free checks, but may not fit a protected arm until R0 has sealed a learned T0 that passes every point utility clause for both seeds and both tasks at the selected ten- or sixteen-bit capacity. The R0 effective seal governs numerical details and takes precedence over this preparatory document.

## Input and public service

All arms inherit each seed's same admitted frozen 16-dimensional purpose latents, fixed non-amplifying scaler, hard-bit mapper and code-only head from the **full OSF_DEFENSE_FIT T0 refit**. Every fitted arm starts from one hash-pinned copy of that T0 checkpoint and receives the same selected bit count, optimizer update plan, task labels and teacher-distillation rule. Each recipient's public release consists of exactly its own hard binary code, probability and decision computed from that code, and constant public model/calibration metadata. There is no clean source score, soft bit, latent or protected-label inference input. The aligned coalition receives both complete recipient views.

The new `code_witness.model.HardCodeModel(..., bits=b)` and `code_witness.learner` are R1-owned. R3 will bind their exact byte hashes and API after they are available. The ten-bit predecessor classes are useful source code, not an admitted new learner or a saved initialization. R3's dynamic kernel and control wrappers live under `code_repair/`; only R3 edits that namespace. R3 will not alter R1's task recipe or shared protocol files.

## Arms and losses

Let `task_i` be the inherited per-purpose true-label cross entropy plus fixed 0.5 teacher KL, and let `m_i` be biased multiscale squared MMD on the actual hard-forward bits of recipient `i`. Let `m_pair` be the biased MMD on the **aligned product kernel**. Use the four inherited Gaussian Hamming-distance scales `gamma=(0.25,0.5,1,2)`. The same one sealed positive `lambda_s` applies to J, L and both sequential orders for a seed. Compute it only by the presealed FIT-only gradient-ratio rule, or use the inherited justified fixed coefficient if R0 seals that choice. No INNER feedback or strength grid.

| Arm | Full-FIT continuation from cloned T0 | Role |
| --- | --- | --- |
| T0 | none; immutable gate-passing checkpoint and calibrator | serious task-only reference and fallback |
| T+ | `(task_1+task_2)/2` | same additional fitting allowance as J |
| L | `(task_1+task_2)/2 + lambda_s*(m_1+m_2)/3` | local-only privacy objective, same per-purpose coefficients as J |
| J | `(task_1+task_2)/2 + lambda_s*(m_1+m_2+m_pair)/3` | kernel candidate |
| S12 | first recipient 1: `task_1/2+lambda_s*m_1/3`; freeze its entire calibrated public release; then recipient 2: `task_2/2+lambda_s*(m_2+m_pair)/3` | 1 then 2 freeze-first design |
| S21 | same with recipient order reversed | 2 then 1 freeze-first design |
| C | `(task_1+task_2)/2 - mean(three balanced critic BCEs)` | ordinary hard-code adversarial control |

For C, the three critics have widths `b,b,2b`, each with `Linear(d,64) -> ReLU -> Linear(64,1)`. Each mapper/head step follows exactly two critic-only Adam 0.001 steps on detached actual hard bits from the same FIT batch. Mapper/head Adam uses the same task learning rate and update plan as T+/L/J/sequential. C has fixed coefficient 1.0. Its extra critic compute is charged and disclosed. Critic weights do not appear on the recipient wire. Its fresh attack is independent of its fitting critics.

For either sequential order, the second stage receives the **saved actual first-stage hard bits on the same FIT people in the same order**. The first mapper, head, calibrated decoder and metadata must be fully frozen before second-stage optimization. Store row-order and first-code hashes privately, and publish their hashes only. Equality between a sequential first stage and a local recipient fit may be registered as an alias only after exact state, calibrator, code and batch-plan identity checks. The first head's calibration must be computed before the second stage and must never change afterward. The second stage must not receive the first person's raw source input, latent or soft bit; it may consume the first released hard bits for its pair training loss under the central fitting arrangement. At deployment, the second recipient's inference remains purpose-local.

## Gate and comparison discipline

R0 must provide a capacity-seal ID, bit count, paired seed IDs, T0 model/scaler/calibrator hashes, exact T0 full-refit source, update horizon and batch-plan identities, calibration source, lambda rule, source latent pins, private package destination and queue contract before R3 can launch a real fit. R3 should reject an absent or inconsistent gate before loading any person-level arrays. There is no automatic substitution of historical W_BANK or continuous F for the learned T0. T0 and eligible T+ each receive fresh complete-view attacks. J can earn a kernel component contrast against T0/T+ only at its measured utility. Competitive and joint claims require the strongest eligible matched controls, including L, both sequential orders and C, with simultaneous inference and no harmful recipient shift. An ineligible J may receive a bounded descriptive attack but cannot earn the full claim.

R4 owns fresh attackers/inference. R5 checks that a frozen first release is the actual first wire output and can independently replay one release and the headline endpoints without R3's trainer/scorer code. R3 does not read HEAD_VALIDATION labels, old-assessment people or all-role arrays.

## Implementation and source limits

The predecessor ten-bit implementation is at `results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1/`. Relevant SHA-256 pins are recorded in `SOURCE_PINS.json`. The old MMD has `PUBLIC_CODE_WIDTH=10`; the old model/head/service and sequential anchor also hard-code ten bits. The new code must derive width from the sealed model and reject mixed widths. The old local/sequential trainer freezes mapper/head weights, but calibrated the first public head only after both sequential stages. The continuation must make that freeze order explicit. The previous C architecture and two-critic-steps rule are applicable, but its model constructor and critic widths need dynamic bit count. No prior numerical result transfers to this changed T0 initialization or sixteen-bit service.

## Count and resource forecast

Per paired seed, conditional fitting entails six new arm identities (T+, L, J, S12, S21, C) and eight mapper/head stage fits: one each for T+, L, J and C, plus two each for S12 and S21. Every arm receives `e_c` FIT epochs per recipient, with `ceil(15434/256)=61` nominal aligned batches per epoch before any deterministic tail merge; the exact presealed plan and counted updates prevail. C adds two critic updates per mapper step. Two seeds therefore have 16 stage fits plus C's critic updates. Calibration requires twelve per-seed global task-temperature fits across the six new arms if no exact alias is admitted; that is 24 across seeds. The two T0 seed packages and their calibration are gate work, not R3 continuation work. R4's complete-view attack and inference counts must be added before the stage seal. This count is an identity, not a wall-time forecast; R0/R3 must use a role-sized smoke timing before launching all arms and keep the 5 CPU-hour conditional-protection reservation and 20 CPU-hour campaign ceiling.

No real protection fit, calibration or attack was run in preparing this document.
