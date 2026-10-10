# Exact incumbent restoration and public action support

P0 is the historical `U|JOINT|i8o64|l0.1|H-CLASS-TEMP` release used by the
bank campaign's P_STAR H2 attacks. Its exact source model, partition, class
temperature decoder, original public coarse IDs and R12 ATT/INNER release
hashes are bound in `P0_SOURCE_MANIFEST.json`. The source anchor is local;
remote verification remains pending. This is a development restoration on
known Adult rows, with no new teacher, partition, head or calibration fit.

## Exact wire and learned inputs

Original bank sizes are seed 0: 14 income and 270 occupation states; seed 1:
13 and 299. Each recipient's source wire has fields `schema`, `recipient`,
`probability`, `decision`, `release_id`, `code_id` and `public_metadata`.
Metadata contains `model_id`, `source_binding_sha256`, `bank_sha256` and
`code_length_bits`. The source code lengths are 4 and 9 bits, not ten. IDs
remain in original table order. No merging, renumbering or metadata removal
is permitted for identity. Both probabilities and decisions are fixed
functions of the original code.

`incumbent_view.r1_deploy.P0Service` applies the exact source raw83 cast,
frozen teacher and task heads, QPC router and H-CLASS-TEMP decoder. The
deployed JSONL reconstructs the actual R12 metadata-bearing wire. Task and
SEX labels, fine-cell IDs, teacher scores, randomness and raw predictors
are absent from the transmitted wire.

The fitted channel receives only P0 source codes and their decoded public
values. Private teacher scores are used by the provider to produce P0 and
for this mechanical historical-space diagnosis. They are not fitted
channel inputs in the new study. Local channels receive only their own
source code and deterministic derived values.

## Fixed action support

For each original P0 source token, the identity action is first, followed
by at most three other same-class P0 public states. Alternatives are ranked
by source symmetric KL with a distance-only 1e-12 floor and renormalization;
ties use the original public global ID. Cartesian products contain at most
16 paired actions. Supports depend only on the original P0 source input
and remain fixed in sequential stage two.

This is a newly instantiated nearest-three neighborhood. The inherited
bank used identity, two nearest states and a class medoid, with another
global-ID order. That rule is not silently relabelled as this rule. No
claim concerns the complete possible release frontier.

## Queued source gate

`R1_P0_S0_ADMISSION` and `R1_P0_S1_ADMISSION` each reserve 180 CPU/wall
seconds and 3 GiB. The active new admission queue guard runs before any
role archive is opened. Every source file and role feature archive is
byte-pinned. The only person-level reads are existing role-only feature
archives for FIT, ATTACK_FIT_NEW and INNER_SELECTION and the source
label-free ATT/INNER P_STAR releases. All archive members are checked
before array materialization. No HEAD, assessment, mixed-role archive,
task-label file or SEX-label file is opened.

The gate routes all admitted rows. Every ATT/INNER code, probability and
decision and the entire metadata-bearing JSON wire hash must equal R12.
On the predeclared 32 equally spaced FIT indices, the actual historical
`hcal.deploy` CLI must agree with the new raw router for both recipients.
The private role output `P0_RELEASE.npz` has exactly `row_id`, `unit`,
`code_r1`, `code_r2`, `prob_r1`, `prob_r2`, `decision_r1`, `decision_r2`.
Hashes, counts and parity status are public; person-level outputs stay
private.

The same gate records duplicate old continuous-score inputs requiring
different P0 outputs and required P0 probability states missing from the
old global bank and per-row bounded support. Complete original wire
identity and probability-state reachability are distinct because old
W_TSTAR had another code and metadata contract. No observed collision
does not establish population learnability. The inherited continuous
scores analytically determine the frozen P0 router; restricted actions
and the old FIT proxy can still exclude P0. R3 owns the independent
utility/proxy cost comparison.

## Independent boundary and custody

R5 independently verifies source bytes, raw routing, actual wires,
reachability and numerical quantities. R1 does not certify its own endpoint
implementation. Later deployment will include P0, selected policy and a
strongest control, with fixed private draws and all public metadata.
Versioned snapshot-only replay must verify emitted wires and representative
reader predictions. Same-device recovery is separate from off-device
backup; predecessor artifacts and failed attempts remain intact.
