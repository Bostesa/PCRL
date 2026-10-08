# Team B prototype implementation receipt

Status: **prefit prototype**. The code implements the mechanism boundary and
loss terms but no Adult fit, held-out calibration fit, fresh attack, or method
comparison. S2 has not verified a role-only source containing just allowed
rows and labels. S3's global compute queue controls all numerical checks.

## Code and release contract

`bottleneck.py` contains two independent 83→64→64→16 Gaussian encoders and
sampled-code task heads; the required numerical log-variance bounds are part of
the model state. `training_terms` computes means over the two task CE terms,
teacher KL terms, diagonal-Gaussian KL terms and local SEX-critic CE terms,
plus the pair SEX-critic CE. `LossWeights` requires all five coefficients. The
caller must implement and lock optimizer steps, critic refreshes and arm-specific
freezing before any actual fit; this module itself performs no optimizer step.
Training critics see sampled features, raw head logits, T=1 probabilities and
one-hot decisions. T=1 is a training surrogate because held-out temperatures
are fitted only after the encoders are frozen.

`ReleaseAdapter` accepts only the declared 83 columns in exact order and
requires a pinned model and schema digest, source identity, Adult class counts
2/6, positive frozen global temperatures, and a private 32-byte-or-longer
sampler key. The custodian must generate that key from a cryptographically
secure source and retain it privately; key bytes and internal person keys never
enter views. HMAC domain separation uses source, model, purpose and person to
make the sampled draw independent of call order. It does not make this a
differential-privacy mechanism. The head consumes the exact sampled `z` that
is transmitted. The recipient sees `z`, raw logits, calibrated probabilities,
decision and metadata. The adapter refuses nonfinite output and never accepts
teacher, task, or SEX labels at inference.

`finalize(path)` requires both recipients' views for each person from the same
input. It writes a private, mode-0600 batch file without replacing an existing
one; `SavedRelease` checks its digest and serves deep copies of frozen views.
The private file contains person-level releases and internal keys and is never
for Git/public results. Serving fresh draws by constructing new adapters is
outside this one-shot contract. The file digest detects accidental change but
is not a keyed authenticity proof against a writer who can rewrite the file.

## Synthetic checks and limitations

`test_bottleneck.py` checks schema/order refusal, sampled-head identity,
temperature application, deterministic ablation, conditional locality,
complete numeric views, KL normalization, objective terms, nonfinite refusal,
model/schema pinning, private-key requirement, aligned immutable replay and
copy isolation. These are engineering checks only; they do not show that a
neural optimizer can solve the frozen laws or Adult task.

`check_synthetic_laws.py` reads S1's frozen `SYNTHETIC_LAWS.json` only after
checking its SHA-256. It enumerates exact rational masses for L1–L5 and checks
the stipulated null, removable, complementary and redundant privacy premises,
plus the confidence advantage in expected NLL and Brier. It does not fit or
select Team B models. The file and laws must not be changed after outcomes.

The two checks are queued with S3. `py_compile` and `git diff --check` are the
only checks run after the queue gate was communicated. Earlier tiny Torch
development-test invocations are disclosed to S3 as an unqueued engineering
deviation and conservatively charged to shared setup.

All seven Adult comparison arms, calibration fitting, training schedule,
fresh attackers and common-criterion reporting still require S1's campaign
lock, S2's safe inputs and S3 admission. This prototype cannot support a
method win, utility preservation or privacy improvement claim by itself.
