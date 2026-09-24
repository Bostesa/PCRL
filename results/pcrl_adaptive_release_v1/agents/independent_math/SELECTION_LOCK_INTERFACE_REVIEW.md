# Selection lock and per-anchor alias review

Read-only review of `selection.py`, `inference.py`, `outer_access.py`,
`outer_audit.py`, `release_specs.py`, and `run_inner.py` at the current adaptive
study source. No outer labels or new candidate outcomes were opened.

## Decision-critical finding

`release_specs.build_release_specs` deduplicates **within each anchor**. Thus a
logical release such as `B_selected` may map to `A_selected` on anchor 0 and to
itself on anchors 1 and 2. A single global mapping from `B_selected` to
`A_selected` would incorrectly erase its latter two comparisons. Selection
should first use `selection.inner_validation_from_report` on each verified
`INNER_AUDIT.json`, then `selection.expand_named_scores` with **that anchor's**
private spec-index aliases. Pass those three named score maps to
`select_route_candidate` and `select_family_representative` under stable logical
names. Neither selector needs the per-anchor alias map once the scores are
expanded. Their `alias_of` argument may be used only for a separately derived
global exact alias relation.

The lock's `alias_of` for `inference.family_manifest` must likewise mean a
global alias among comparator *logical names*: compare the three-element
canonical-ID vectors for each name and collapse only if the entire vector is
identical. Choose a stable representative, preferring `D17` within its exact
alias group, then lexical ID. Preserve all original names in
`comparator_names`. Never collapse a release that aliases on one or two anchors
only. A paired difference is exactly zero on an aliased anchor but can be
nonzero elsewhere. `H` is a distinct reference, not a token-law alias.

## Proposed exact lock fields

The existing required fields remain: `schema`, `status=LOCKED`, `study`,
`assessment_year=2018`, `outer_assessment_authorized`, `protocol_sha256`,
`input_index_sha256`, `slots`, `alias_of`, `family_manifest`, and all three
`anchors`. Freeze these additional records before any outer read:

* `selection_source`: validation-only source (`inner_selection`), hashes of
  `selection.py` and `inference.py`, deterministic candidate-family memberships,
  ranking outputs for U/P candidates and every mandatory independent baseline
  family, point-screen/diagnostic status, and the selected logical IDs. A
  builder should extract only `candidate_validation_scores` and
  `H_validation_scores`; `candidate`, `H`, and private contributions in the inner
  report are `inner_check` outcomes and must not enter nomination or representative
  selection.
* `slots`: at most one U and one P slot. Each has a unique slot ID, arm, logical
  candidate release ID, nomination/diagnostic status, and comparator **logical**
  IDs including D17 and every mandatory family representative. Current
  `inference.enumerate_endpoints` treats slot `id` as the score key; if U and P
  choose the same physical release, use two distinct slot score IDs mapped to
  the one physical release, or record one slot with an explicit protocol choice.
  Do not describe the duplicate as two independent algorithmic successes.
* `family_manifest`: exact output of `inference.family_manifest(slots,
  alias_of=global_exact_aliases)`, plus its hash if stored separately. Include
  all required roles and both weightings under the frozen U/P signs and
  thresholds. `capability_endpoints` can remain separately reported because
  the H benefit is a point-estimate screen with separate uncertainty.
* `anchors[a]`: pinned full `inner_panel_complete_sha256` and
  `inner_audit_sha256`; pinned private spec-index and binding SHA-256; the full
  logical-to-canonical alias map from that spec index (including
  `D17 -> aliases['A_control_D17']`); and `releases`, the canonical IDs to be
  scored with their exact inner `source` descriptors. Those canonical IDs are
  the union of each locked endpoint's plus/minus logical release after the
  anchor mapping, excluding special `H`. Preserve full-panel hashes even when
  this scoring union is a subset of the inner panel.

Validation before outer unlock should require: (1) every endpoint plus/minus
and every slot/comparator logical ID resolves on **all three anchors**;
(2) each resolved non-H canonical ID occurs in `anchors[a].releases` and the
pinned full inner report; (3) the saved descriptor equals the inner descriptor;
(4) the per-anchor alias map matches the pinned spec index and binding; (5)
the global `alias_of` relation is exactly the three-anchor vector equivalence;
(6) every mandatory comparator family and D17 is represented; and (7) the
reconstructed endpoint manifest equals the frozen one. Then outer scoring may
load each distinct canonical law once per anchor and copy its *same stored
person-level score record* to the logical aliases for inference. This copying
does not create additional households or fitted predictors. The full inner
receipt inventory and selected model hashes remain verified before the label
gate, as `outer_audit` currently does for scored canonical releases.

## Current interface gaps and regression cases

`outer_access.verify_outer_unlock` currently checks the manifest against slots
and pins an anchor `releases` dictionary, but does not itself establish the
per-anchor logical-to-canonical mapping or prove every endpoint resolves to a
scored release. `outer_audit.score_locked_outer` does compare the passed
canonical subset to that anchor's locked `releases`, checks full inner report
and COMPLETE hashes/inventory, verifies scored descriptors and saved routes,
then opens outer labels. The missing cross-anchor mapping/required-family
checks belong in lock construction and in the pre-unlock verifier, not in
post-outcome inference.

Use synthetic negative fixtures where: `B_selected` aliases A only on anchor
0; a lock endpoint refers to an unmapped logical name; an anchor alias points
to a canonical ID outside the locked scored subset; a selected comparator
family is absent; and extreme `inner_check` losses disagree with unchanged
validation scores. All must fail or retain the same selection **before**
`outer_access.load_locked_outer_role`. Also verify that a global exact alias
collapses endpoints once while keeping every declared comparator name.
