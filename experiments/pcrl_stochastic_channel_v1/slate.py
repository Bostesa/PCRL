"""Routing untouched-J predictors into an extended release's own candidate slate.

The predecessor (`pcrl_utility_extension_v1`) promised in METHOD section 1 that "untouched-J
predictor candidates are part of every comparison", justified by the fact that a larger input can
make a *finite* learner worse even though it cannot raise the Bayes risk. What it implemented was
weaker: `acs_spectral_audits.route_ancestor` routes **H-only** ancestors (columns
`0..H_A_WIDTH-1`), and `ref_J` is audited as a separate *condition* whose gains are differenced at
the metric level in `program.screen`. Those two numbers come from independently selected slates, so
nothing in the extension's own slate recovers a J-only predictor that the wider input handicapped.

This module supplies the missing candidates. The distinction it maintains:

* **J comparator** -- `ref_J`, a separate audited condition. Already present. Unchanged.
* **J anchor** -- a predictor fitted on the untouched-J wire, evaluated *inside* an extended
  release's slate by reading only the `[H_A, Z_J]` columns and ignoring `R`. Added here.

Effect on measurement, registered in advance: selection is `argmin` validation log loss
(`acs_fixed_predictions_audits.select_pools`), so adding candidates can only lower the selected
attacker loss, i.e. **raise** measured recovery and **raise** the increment over J. This correction
makes the protection screen strictly harder to pass. It is not a repair, and no historical result
is re-derived from it.
"""
from __future__ import annotations

import copy

from experiments import acs_fixed_predictions_audits as old
from experiments.pcrl_direct_adversarial_v1 import inputs as dax

# The untouched-J wire is `wire/A = [H_A, Z_J]`; an extended wire appends `R` after it.
J_BASE = dax.H_A_WIDTH + dax.A0_WIDTH


def route_j_anchor(view: str, width_a: int, columns=None, *, base: int = J_BASE,
                   hb_width: int = dax.H_B_WIDTH) -> tuple:
    """Columns of an extended wire that reproduce the untouched-J wire, bit for bit.

    `A`  -> `[H_A, Z_J]`      = `0 .. base-1`, skipping the whole `R` block.
    `AB` -> `[H_A, Z_J, H_B]` = the same, then `H_B` at `width_a, width_a+1` (`build_wires`
            appends `B` last, so `H_B` sits after the extension, not at a fixed offset).

    `columns` is the projection the source candidate already used on its own wire; it is composed
    with the route, exactly as `acs_spectral_audits.route_ancestor` composes for H ancestors.
    """
    if view == 'A':
        route = tuple(range(base))
    elif view == 'AB':
        route = tuple(range(base)) + tuple(range(width_a, width_a + hb_width))
    elif view == 'B':
        raise ValueError('the B wire is H_B in every system; canonical B candidates are shared '
                         'unchanged, so a J anchor there would only duplicate them')
    else:
        raise ValueError(f'unknown view {view!r}')
    return tuple(route[j] for j in columns) if columns is not None else route


def j_anchor_metadata(source_meta: dict, ident: str, view: str, columns) -> dict:
    """Metadata for an in-slate J predictor, marked so it is never confused with an H ancestor."""
    meta = copy.deepcopy(source_meta)
    meta.update(candidate_id=ident, view=view, projection_columns=list(columns),
                source_condition='ref_J', j_anchor=True, ignores_extension=True,
                anchor_ancestor=False, inherited_singleton=False,
                provenance=('fitted on the untouched-J wire and routed into this release; reads '
                            '[H_A, Z_J] only and ignores the extension block'))
    return meta


def build_j_anchor_candidates(j_audits: dict, width_a: int) -> dict:
    """The `extra_candidates` supplement for `acs_spectral_audits.build_audits`.

    Shaped `{budget: {role: {candidate_id: AuditCandidate}}}`, covering the A and AB roles only.
    `build_audits` merges it before `inherit_singletons`, so these anchors are also inherited into
    the coalition roles, and its existing route-parity assertion re-scores every one of them and
    checks the recomputed validation metrics against the scores recorded when the candidate was
    fitted on the untouched-J wire. That assertion is the routing's correctness check: it can only
    pass if the selected columns reproduce the J wire bit for bit.
    """
    out = {budget: {role: {} for role in roles if role.split('/')[0] != 'B'}
           for budget, roles in j_audits['candidates'].items()}
    add_j_anchors(out, j_audits, width_a)
    return out


def add_j_anchors(candidates: dict, j_audits: dict, width_a: int) -> list:
    """Insert ignore-R untouched-J predictors into every A and AB role slate, in place.

    Must run before `acs_fixed_predictions_audits.inherit_singletons`, so the routed anchors are
    themselves inherited into the coalition roles. Returns the list of `[budget, role, id]` added.

    Skipped, to avoid duplicating candidates the slate already gets by another route:
    * `B` roles -- canonical B candidates are shared unchanged in every system;
    * the source's own H ancestors -- the extension routes those itself;
    * the source's inherited singletons -- `inherit_singletons` regenerates them here.
    """
    added = []
    for budget, roles in candidates.items():
        for role, current in roles.items():
            view = role.split('/')[0]
            if view == 'B':
                continue
            for cid, candidate in j_audits['candidates'][budget][role].items():
                meta = candidate.metadata
                if meta.get('anchor_ancestor') or meta.get('j_anchor') or cid.startswith('inherited_'):
                    continue
                cols = route_j_anchor(view, width_a, candidate.columns)
                ident = 'j_anchor__' + cid
                current[ident] = old.AuditCandidate(
                    candidate.base, 'wire', cols, j_anchor_metadata(meta, ident, view, cols))
                added.append([budget, role, ident])
    return added
