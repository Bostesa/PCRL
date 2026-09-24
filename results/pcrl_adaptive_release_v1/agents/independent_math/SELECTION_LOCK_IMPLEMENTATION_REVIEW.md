# Independent selection-lock implementation review

Read-only review of the new `selection_lock.py` and
`outer_access.verify_lock_structure`, with a separate synthetic fixture at
`tests/pcrl_adaptive_release_v1/test_selection_lock_review.py`. I did not
dispatch a scientific unit or open 2018 person rows.

**Pre-dispatch verdict:** no blocker in the four requested lock invariants.

* `_verified_anchor` checks each full inner panel's COMPLETE inventory,
  report, spec index, and binding hashes, along with the registered standard
  slate and inner roles. It calls `selection.inner_validation_from_report`,
  which reads the validation-selected route scores rather than `candidate` or
  private `inner_check` contributions. The independent fixture builds two
  complete, hash-consistent panels with reversed extreme inner-check losses;
  the lock input scores remain identical.
* `global_exact_aliases` compares every named law's canonical identity vector
  over anchors 0, 1, and 2. An alias on only one anchor stays a distinct
  global comparison. The builder selects from each anchor's expanded logical
  names and constructs unique U/P slot score keys mapped to the selected
  physical release.
* Simple, task-only, gradient, and constrained deterministic families must
  have named fitted points. The D17 fallback is included in each family/route
  selection, and `verify_lock_structure` requires its selected representative
  and D17 in every slot.
* `verify_lock_structure` reconstructs the frozen endpoint family, checks
  declared global aliases agree on all anchors, and requires every endpoint,
  comparator and slot logical name to map to a canonical ID in that anchor's
  locked scoring subset. The outer scorer separately verifies exact lock and
  full inner COMPLETE/report hashes, full artifact inventory, frozen release
  descriptors and selected model hashes before calling the outer-label loader.

The three private B-center archives were read back by their **recorded S3
versions**, each AES256 encrypted. Their tarball SHA-256s match `RUN_STATE.json`:
anchor 0 `0c319e0e88f9b02925087d0f648ad93f316f76268fd79c94cf581e2db932fd65`,
anchor 1 `cc681368f163632833728daf5c5b4d3cf46ed10a83e60ceff4229cf7c6d4b201`,
anchor 2 `1e2ff6b1d7a5a97489799ec2d9b732b5e7056d7dcd703dc21f29cdece69417f4`.
Each COMPLETE hash matches the ledger and pins its corresponding A COMPLETE.
Every B receipt says `SUPPORT_LIMITED_ALIAS_A` and `realized_states=32`; each
partition has zero rules and identity parents 0..31. The archived 32×17 B
channel's shape/dtype/bytes hash equals that receipt's
`a_selected_channel_array_sha256` on each anchor. There is no distinct B
release in this center panel, so it cannot support adaptive-split component
attribution. The B branch must remain visible as a support-limited failed
branch, with no B-specific gain claimed.

Two checks should remain explicit in the coordinator's lock review. First,
`verify_lock_structure` is a structural verifier: it does not independently
rehash the private spec index or recompute its alias map. The builder checks
the private binding, and the outer scorer checks the frozen full panel before
labels, so use this builder's output and keep the spec/binding SHA pins in the
lock. Second, the generic endpoint enumerator cannot know that a family name
represents a strong independently constructed comparator; retain the builder's
family roster/selection evidence and report that the B branch made no new
physical law. If U and P slots map to the same A object, they are two
predeclared route tests of **one** trained release, not two independent wins.

Focused suites: `test_selection_lock_review.py`, `test_selection_lock.py`, and
`test_outer_access.py` passed 6/6. The new independent fixture verifies the
completed-panel validation/check separation through actual receipt hashes.
