# Designer B, fit report at 15:26Z (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:26Z on 2026-10-20.
Saved verbatim; nothing below was edited. The fit-time diagnostics are in-sample MODEL_FIT readings; verdicts use MODEL_SELECT through
nrm.evaluate, nrm.screen and the grid.

---

DESIGNER B (15:26Z): all 18 family-B releases are saved and ready for nrm.evaluate and nrm.screen. No sub-agents were used after the rule (still the 2 read-only Explore agents from about 14:37-14:45Z, both finished). Everything ran on one thread.

GO changes, all done before the fit:
1. PCA is dropped. MINF is added: the same rank-1 coupled construction along each person's minimum-Fisher eigenvector, with unit Fisher norm, the same s, 1.5 cap, R = 8, key scheme and channel-aware training. FB_ISO_S2 is added: full-rank, uncoupled, Fisher-isotropic.
   - MINF has its own selftest checks, all passing: unit Fisher norm, minimum-eigenvalue direction, realized cost 0.00492 against s²/2 = 0.005 on invented data, rank-1 shared keyed draw, and emit_one equals emit.
2. F1 is restated against MINF at S2 and S3, with ISO reported beside. The declared twin is FB_TWIN; the simplest alternatives are MINF (the G3 path), ISO, ID, MEAN and POSTHOC.
3. The card states:
   - deployed randomness for people outside the admitted rows: per-request keyed draws from a service CSPRNG key, one instance per person, repeated-query protection not claimed;
   - custody of the root key: private only, replay from the private tree or backup, verifier replay allowed;
   - SEX-labelled exposure: δ and Σ_w are a closed-form LDA-type critic on in-sample logits, counted as n_critic_fits in each record;
   - per-task spend is read beside the linked cost;
   - no coupling or joint-design claim.
   - MINF's spend generically falls on one recipient, and the cap lowers MINF's realized cost.

Commits (my paths only):
- e4da5a8fc: code. nrm/fb/channel.py and learner.py; selftests channel 27/27, learner 20/20.
- 48fbac50f: card results/pcrl_new_release_mechanisms_v1/cards/FB_FSC_v1.md, SHA-256 472da932…, committed 15:24:05Z.

Fit units (shared queue, stage fit, rss 2 GiB, torch interpreter):
- FB_FIT_V0: exit 0, 28.3 CPU-s, 9 configurations saved.
- FB_FIT_V1: exit 0, 27.6 CPU-s, 9 configurations saved.
- Releases: runs/FB/<config>/V<v>/. Model files: runs/FB/MODELS/V<v>/.
- Configurations: FB_TWIN, FB_SEX_S2, FB_SEX_S3, FB_MINF_S2, FB_MINF_S3, FB_ISO_S2, FB_SEX_S2_ID, FB_SEX_S2_MEAN, FB_POSTHOC_S2, each in V0 and V1. The ledger shows 18 FB entries.
- Randomized configurations (SEX, MINF, ISO, POSTHOC) are (8, n, k) stacks. TWIN, ID and MEAN are deterministic (n, k).

Label reads (4, all with the card):
- FB_FIT_V1 at 15:24:24Z and FB_FIT_V0 at 15:24:25Z, each reading:
  - MODEL_FIT, operation 'fit': income, occupation, sex (20,000 people);
  - CALIBRATION, operation 'calibrate': income, occupation (2,982 people).
- MODEL_SELECT was not read.

Deployment check (label-free, out of queue, charged): nrm.fb.release.predict reproduces draw 0 bitwise for all 18 configurations × 6 roles, using keys {'role': R, 'row_id': row ids of R}.

Training: early stopping selected 7 epochs in V0 and 6 in V1 for every learner (paired). The CALIBRATION temperatures are about 1.18-1.25 for income and 0.92-1.05 for occupation.

FIT-TIME DIAGNOSTICS (MODEL_FIT, in-sample, recorded in each RELEASE.json; not evidence for any verdict, which uses MODEL_SELECT through nrm.evaluate and nrm.screen):

Emitted NLL relative to FB_TWIN, mean over R = 8 (income / occupation, nats):

| Configuration | V0 | V1 |
|---|---|---|
| SEX_S2 | +0.0005 / +0.0043 | +0.0004 / +0.0030 |
| SEX_S3 | +0.0016 / +0.0072 | +0.0018 / +0.0058 |
| MINF_S2 | +0.0005 / +0.0055 | +0.0003 / +0.0035 |
| MINF_S3 | +0.0009 / +0.0086 | +0.0016 / +0.0085 |
| ISO_S2 | +0.0010 / +0.0037 | +0.0008 / +0.0034 |
| POSTHOC_S2 | +0.0007 / +0.0034 | +0.0009 / +0.0032 |
| ID | about 0 / +0.0010 | about 0 / about 0 |
| MEAN | about 0 / +0.0010 | about 0 / about 0 |

- Most of the spend falls on occupation.

Linked linear SEX discriminability Δ², one draw:

| Configuration | V0 | V1 |
|---|---|---|
| TWIN | 0.249 | 0.248 |
| SEX_S2 | 0.203 | 0.209 |
| SEX_S3 | 0.184 | 0.194 |
| MINF_S2 | 0.247 | 0.241 |
| MINF_S3 | 0.248 | 0.239 |
| ISO_S2 | 0.236 | 0.232 |
| ID | 0.251 | 0.248 |
| MEAN | 0.250 | 0.250 |
| POSTHOC_S2 | 0.207 | 0.199 |

- At fit time the SEX channel removes 16-26% of the linear discriminability, while MINF removes about 0-3%.
- POSTHOC matches SEX_S2 at no higher cost. That leans toward P5 (joint adaptation adds little). It is a fit-time reading only.
- The fall at S2 is under 50%, so F4's trigger cannot fire on these numbers.

CPU charged out of queue under label designer_b: 40 + 25 + 10 s, all upper estimates. Queue units: about 56 CPU-s.

Next: waiting for your nrm.evaluate (utility; P1 twin gate, per-task spend) and nrm.screen outputs. No further fits until then.
