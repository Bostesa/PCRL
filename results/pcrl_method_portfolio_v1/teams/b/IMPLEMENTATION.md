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
caller must lock optimizer steps, critic refreshes and arm-specific freezing
before any actual fit; this module itself performs no optimizer step.
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

`train_bottleneck.py` now supplies an explicit stage runner. It requires
caller-provided tensors and ordered private row/group IDs from one admitted
fitting role, the exact ordered S2 permitted-column manifest and its pinned
Team B schema hash, exact Ucal teacher identities, an explicit objective and training
schedule, and a stage ID. It updates the declared local and coalition critics,
then the active encoder/head on each step. Its one-purpose objective excludes
the inactive task. Separate named data, encoder and critic generators make
encoder sampling independent of a control's critic-update count. Versioned
private checkpoints contain optimizer and generator states for exact resume.
The sequential second-stage runner requires an aligned, fixed first-stage
one-draw feature artifact tied to its model, checkpoint, row/group order,
schema and sampler-plan hashes; it never re-draws that first recipient.
The Team B schema hash uses SHA-256 of compact JSON encoding of the ordered
names. S2's inherited source hash uses SHA-256 of newline-joined names; the
two algorithms and digests must both be recorded and never interchanged.
One-purpose LOCAL and sequential first stages evaluate only their active
encoder, so their actual forward count matches the unit plan.
`test_train_bottleneck.py` checks these invariants on synthetic tensors.
`time_train_bottleneck.py` is a full-size **synthetic-only** 600-step joint
training timing unit for S3's queue; it contains no Adult input path.

`check_synthetic_laws.py` reads S1's frozen `SYNTHETIC_LAWS.json` only after
checking its SHA-256. It enumerates exact rational masses for L1–L5 and checks
the stipulated null, removable, complementary and redundant privacy premises,
plus the confidence advantage in expected NLL and Brier. It does not fit or
select Team B models. The file and laws must not be changed after outcomes.

Numerical checks and the timing unit await S3 admission and current source-hash
registration. Static compilation and diff checks do not establish numerical
correctness. Earlier tiny Torch development-test invocations are disclosed to
S3 as an unqueued engineering deviation and conservatively charged to shared
setup.

All eight Adult comparison arms, calibration fitting, training schedule,
fresh attackers and common-criterion reporting still require S1's campaign
lock, S2's safe inputs and S3 admission. This prototype cannot support a
method win, utility preservation or privacy improvement claim by itself.
