# Incumbent and selected-channel deployment contract

The provider holds the permitted 83-column input, the frozen U teacher,
heads and P0 partition/router. `r1_deploy.P0Service` produces original P0
public code IDs with the historical H-CLASS-TEMP table. The fitted policy
receives only those original source IDs and their decoded P0 probabilities;
it cannot inspect raw columns, frozen teacher scores, task labels or SEX.
The public checkpoint and action bank are available to defense-aware readers.

`r1_deploy` emits one recipient JSONL file with the historical P0 schema.
`r1_selected_deploy` emits both recipient JSONL files. A selected output
uses an original P0 public ID, its fixed probability and decision, and the
source's unchanged `release_id`, `code_length_bits`, bank and source-binding
metadata. No arm, sampler branch, fine token, raw score, coin or fallback
indicator is added to a person's wire. The selected policy package and
its hash-bound receipt remain public method metadata. A complete-view
attack may use every transmitted field and that public policy.

The selected CLI requires a byte-pinned private source manifest, policy
receipt, scientific lock SHA, arm ID, seed and raw input archive. Stochastic
arms also require a byte-pinned owner-only `uniform.npy` archive. Joint
channels consume one float64 uniform per person. Product, local and both
sequential channels consume two recipient-domain float64 uniforms per
person. P0 and deterministic controls accept no coin file. Values must be
finite in `[0,1)`, with exact dimensions. One fixed output per person is
the declared inference contract; repeated-query protection is not claimed.
The policy must use the exact original probability tables and fixed
identity-plus-nearest-three same-class supports. The adapter rejects a
changed decision, altered codebook or unregistered law kind.

The queued `r1_replay` worker will derive realization-0 coins from the
inherited private HMAC stream root and FIT group IDs, run the actual CLI on
the frozen 32 raw83 FIT rows, and compare both complete JSONL wires with
the selected policy's cached FIT release. No label archive is an input.
R5 separately verifies source identity, output hashes and selected replay.
The runnable raw CLI and a successful replay are distinct states; no
selected-policy numerical replay has occurred yet.
