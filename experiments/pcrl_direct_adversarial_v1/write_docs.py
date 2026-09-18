"""Stage 10: generate the data-backed deliverables from saved artifacts.

Fits nothing, selects nothing, decides nothing. Every table is generated from a file on
disk and every document names the file it came from, so a table and its machine-readable
backing can never drift apart.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np

from .inputs import OUT, read_json, sha_file, write_json
from .report import (BETAS, CATCHUP_SCOPE, FAMILY_SENSITIVE, MAIN, MAIN_SCOPE, NEW_ARMS,
                     NO_PROTECTION, POLICIES, SINGLE, WIDTHS, main_arm)

FAMILY_ENDPOINT_LABEL = {
    'recovery/A/SEX': 'A/SEX', 'recovery/AB/SEX': 'AB/SEX',
    'recovery/A/RAC1P': 'A/RAC1P', 'recovery/AB/RAC1P': 'AB/RAC1P',
    'utility/same_residence': 'residence loss',
}


def rows_of(path: Path) -> list:
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def fnum(value, places=4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return '--'
    return f'{x:+.{places}f}'


def table(header, rows) -> str:
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


# ------------------------------------------------------------------ baseline completion
def baseline_transport(out: Path) -> str:
    proofs = {}
    optnet = out / 'OPTNET_RECONSTRUCTION.json'
    if optnet.exists():
        proofs['optnet'] = read_json(optnet)['proofs']
    fit = out / 'exploratory_2017' / 'EXPLORATORY_2017_FIT.json'
    if fit.exists():
        proofs['all'] = read_json(fit).get('identity_proofs', {})

    lines = ['# BASELINE_TRANSPORT_COMPLETION — the 15 missing 2017 interfaces', '',
             'The `leace_A0`, `splince_A0` and `optnet16_{L1,L2,C1}` adaptations of commit',
             '`73903b7f28df68284285f0610a4036beb32b208f` were never evaluated on 2017, so the',
             "previous study's strongest finding — that a closed-form 2023 linear eraser matches",
             '`J` and beats four studies of developed mechanism — had been seen on **one year',
             'only**. This completes those **15 interfaces** (5 arms x 3 seeds).', '',
             '**These are EXPLORATORY CROSS-YEAR DEVELOPMENT numbers.** The 2017',
             '`final_evaluation` partition is spent. The original frozen 2017 transport result',
             'keeps its historical status and is neither restated nor overwritten. 2017 is not a',
             'new test merely because these methods had not been scored there.', '',
             '## Transport rule', '',
             'The **2018-fitted transformation is reused** on 2017. No new 2017 eraser or',
             'encoder is fitted and then called transported.', '',
             '* `leace_A0` / `splince_A0`: the frozen `A0` inference pipeline is run on the 2017',
             '  features through the transport study\'s own `FrozenSeed.interface`, and the',
             '  **saved affine map is applied to that auxiliary channel alone**. `H_A` and `H_B`',
             '  are appended unchanged.',
             '* `optnet16_*`: the saved encoders act on the frozen whitened features, exactly as',
             '  on 2018.', '',
             '## Identity proofs', '',
             'Neither the erasure affine maps nor the OptNet encoders were persisted by the',
             'invariant study\'s production run. Both stages are deterministic, so both are',
             '**reconstructed** — and neither is permitted to inherit the old identity until it',
             'is proved. The proof used is the strongest available: the reconstruction rebuilds',
             'the **2018** release and must match the stored 2018 `releases.npz` **bitwise on**',
             '**all seven pools**. A mismatch would be recorded as a new fit under a new name.',
             '']
    rows = []
    merged = defaultdict(dict)
    for source in proofs.values():
        for seed, per_arm in source.items():
            for arm, proof in per_arm.items():
                merged[(seed, arm)] = proof
    for (seed, arm), proof in sorted(merged.items()):
        rows.append([seed, f'`{arm}`', proof.get('status', '--'),
                     f"{proof.get('max_abs_difference', float('nan')):.3e}"
                     if proof.get('max_abs_difference') is not None else '--',
                     'yes' if proof.get('identity_proved') else 'no'])
    if rows:
        lines += [table(['seed', 'arm', 'status', 'max abs difference vs stored 2018',
                         'identity proved'], rows), '']
    else:
        lines += ['_No identity proof has been written yet._', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ 2018 development
def development_2018(out: Path) -> str:
    record = read_json(out / 'DEVELOPMENT_2018.json') if (out / 'DEVELOPMENT_2018.json').exists() \
        else None
    per_seed = rows_of(out / 'PER_SEED.csv')
    intervals = rows_of(out / 'PAIRED_INTERVALS.csv')

    lines = ['# DEVELOPMENT_2018 — every configuration and every seed', '',
             '**DEVELOPMENT EVALUATION on 2018 pools that this project has used repeatedly.**',
             'The paired household bootstrap quantifies sampling variability for **fixed fitted',
             'systems**. It cannot undo repeated use, and it is neither Census replicate-weight',
             'variance nor retraining variability. Three anchor seeds do not establish broad',
             'training-population robustness.', '']
    if record:
        lines += [
            f"Conditions scored: **{len(record['conditions'])}** "
            f"({len(record['new_arms_scored_here'])} new, "
            f"{len(record['reused_never_rescored'])} reused and never rescored). "
            f"Contrasts run: **{record['contrasts_run']}** of {record['contrasts_planned']} "
            'planned.', '',
            f"Bootstrap: {record['bootstrap']['replicates']} replicates over "
            f"{record['bootstrap']['clusters']} cohort households. Within-contrast adjustment: "
            'studentized max-|t| across the five family endpoints. Candidate-wide adjustment: '
            'studentized Bonferroni over the whole searched family.', '',
            f"Score replay: {record['score_replay']['checks']} checks, max abs difference "
            f"{record['score_replay']['max_abs_difference']:.2e}.", '']
        if record['missing_conditions']:
            lines += ['**Conditions absent from every table** (not silently omitted): '
                      + ', '.join(f'`{c}`' for c in record['missing_conditions']), '']

    # Seed-mean additional recovery over H, every new arm, unweighted and person-weighted.
    for weight in ('unweighted', 'person_weighted'):
        grouped = defaultdict(list)
        for row in per_seed:
            if (row['kind'] == 'additional_recovery' and row['weight'] == weight
                    and row['scope'] == MAIN_SCOPE and row['budget'] == '360'
                    and row['endpoint'] in FAMILY_SENSITIVE):
                grouped[row['condition'], row['endpoint']].append(float(row['value']))
            if (row['kind'] == 'utility_gain_vs_H' and row['weight'] == weight
                    and row['scope'] == MAIN_SCOPE and row['budget'] == '360'
                    and row['endpoint'] == 'same_residence'):
                grouped[row['condition'], 'residence_gain'].append(float(row['value']))
        conditions = sorted({c for c, _ in grouped})
        if not conditions:
            continue
        body = []
        for condition in conditions:
            cells = [fnum(np.mean(grouped[condition, e])) if (condition, e) in grouped else '--'
                     for e in FAMILY_SENSITIVE]
            residence = (fnum(np.mean(grouped[condition, 'residence_gain']))
                         if (condition, 'residence_gain') in grouped else '--')
            body.append([f'`{condition}`', *cells, residence])
        lines += [f'## Seed-mean additional recovery over `H`, {weight}, '
                  'scope `kernel_expanded_independent` (matched exposure), budget 360', '',
                  'Lower recovery is less disclosure. Residence gain is a capability, so higher',
                  'is more. **Observed negative increments are reported as measured and are**',
                  '**never truncated to zero.**', '',
                  table(['condition', *(e.split('/', 1)[1] for e in FAMILY_SENSITIVE),
                         'residence gain'], body), '']

    if intervals:
        lines += ['## Adjusted simultaneous intervals, by comparison family', '',
                  'Negative means the left arm leaks **less**; for residence, positive means the',
                  'left arm **costs more** utility. `within` is the reused within-contrast',
                  'max-|t| bound; `candidate-wide` is the studentized Bonferroni bound over the',
                  f"whole searched family (m = {intervals[0].get('family_size', '--')}).", '']
        by_family = defaultdict(list)
        for row in intervals:
            by_family[row['comparison_family']].append(row)
        for family, rows in sorted(by_family.items()):
            significant = [r for r in rows
                           if r['significantly_better'] == 'True'
                           or r['significantly_worse'] == 'True'
                           or r['candidate_wide_better'] == 'True'
                           or r['candidate_wide_worse'] == 'True']
            lines += [f'### `{family}` — {len(rows)} rows, '
                      f'{len(significant)} with an interval excluding zero', '']
            if significant:
                body = [[f"`{r['left']}` vs `{r['right']}`", r['weight'],
                         FAMILY_ENDPOINT_LABEL.get(r['endpoint'], r['endpoint']),
                         fnum(r['estimate']),
                         f"[{fnum(r['adjusted_low'])}, {fnum(r['adjusted_high'])}]",
                         f"[{fnum(r['candidate_wide_low'])}, {fnum(r['candidate_wide_high'])}]"]
                        for r in significant[:40]]
                lines += [table(['contrast', 'weight', 'endpoint', 'estimate', 'within',
                                 'candidate-wide'], body), '']
                if len(significant) > 40:
                    lines += [f'_({len(significant) - 40} further significant rows in '
                              '`PAIRED_INTERVALS.csv`.)_', '']
            else:
                lines += ['No interval in this family excludes zero under either adjustment.', '']

    lines += ['## Backing data', '',
              '* `PER_SEED.csv` — every condition, seed, weighting, budget and scope',
              '* `PAIRED_INTERVALS.csv` — every contrast, both adjustments',
              '* `PER_SEED_SIGNS.csv` — per-seed effect sizes and signs',
              '* `SCORE_REPLAY.csv` — stored versus recomputed endpoint values',
              '* `DEVELOPMENT_2018.json` — decisions, registered rules and the input registry', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ 2017 panel
def exploratory_2017(out: Path) -> str:
    path = out / 'EXPLORATORY_2017.json'
    lines = ['# EXPLORATORY_2017 — the fixed cross-year panel', '',
             '**Every number on this page is EXPLORATORY DEVELOPMENT.** The 2017 locked',
             'evaluation is finished and its `final_evaluation` partition is spent. This is not',
             'a second confirmation of anything. The original frozen 2017 transport result keeps',
             'its historical status and is neither restated nor overwritten. **2017 is not a new',
             'test merely because these methods had not been scored there.**', '',
             'Panel membership was fixed in `PROTOCOL.md` §10 **before any 2017 number was',
             'read**, so nothing here was selected by its residence result.', '',
             'No intervals are computed. Were any computed they would be **conditional',
             'descriptive sampling uncertainty**, which does not restore independence after',
             'repeated development — and which is not mathematically forbidden on a used',
             'dataset either.', '']
    if not path.exists():
        return '\n'.join(lines + ['_The 2017 panel has not been scored yet._', '']) + '\n'
    record = read_json(path)
    lines += [f"Panel declared: {len(record['panel']['main'])} main + "
              f"{len(record['panel']['no_protection'])} no-protection + "
              f"{len(record['panel']['new_erasure'])} new erasure + "
              f"{len(record['panel']['historical_baselines'])} historical baselines, "
              'x 3 seeds.',
              f"Conditions present: **{len(record['conditions_present'])}**. "
              f"Missing units: **{len(record['missing_units'])}**.", '']
    if record['missing_units']:
        names = sorted({m['condition'] for m in record['missing_units']})
        lines += ['**Missing, listed rather than omitted:** '
                  + ', '.join(f'`{n}`' for n in names), '']
    for weight in ('unweighted', 'person_weighted'):
        grouped = {}
        for row in record['summary']:
            if row['weight'] != weight:
                continue
            grouped[row['condition'], row['kind'], row['endpoint']] = row['seed_mean']
        conditions = sorted({c for (c, _k, _e) in grouped})
        if not conditions:
            continue
        body = []
        for condition in conditions:
            cells = [fnum(grouped.get((condition, 'additional_recovery', e)))
                     for e in FAMILY_SENSITIVE]
            residence = fnum(grouped.get((condition, 'utility_gain_vs_H', 'same_residence')))
            body.append([f'`{condition}`', *cells, residence])
        lines += [f'## Seed means, scope `common_fresh`, budget 360, {weight}', '',
                  table(['condition', *(e.split('/', 1)[1] for e in FAMILY_SENSITIVE),
                         'residence gain'], body), '']
    lines += ['## Backing data', '', '* `EXPLORATORY_2017.csv` — every row',
              '* `EXPLORATORY_2017.json` — panel declaration, missing units, input registry', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ attack strength
def attack_strength(out: Path) -> str:
    mech = read_json(out / 'MECHANISM.json') if (out / 'MECHANISM.json').exists() else {}
    catchup = rows_of(out / 'MECH_CATCHUP.csv')
    gap = rows_of(out / 'MECH_TRAINING_VS_AUDIT.csv')
    refresh = rows_of(out / 'MECH_REFRESH.csv')

    lines = ['# ATTACK_STRENGTH — what the attacks actually found', '',
             'A low measured recovery means the **declared finite attack family** did not find',
             'one inside its budget. It is not independence, not differential privacy, not a',
             'bound on `I(S;Z|H)` and not protection against arbitrary attackers.', '',
             '## Training attacker versus fresh auditor', '',
             'Two different quantities on two different pools: the training gain is measured on',
             'the internal `monitor` fold against a 300-update differentiable probe; the audit',
             'recovery is measured on the test pool against the full independent slate, which',
             'includes boosted trees and kernel ridge that were **never in the training',
             'gradient at all**. A large gap means the mapper was fooling a family the auditor',
             'does not belong to.', '']
    if mech.get('training_versus_auditor'):
        block = mech['training_versus_auditor']
        lines += [f"Rows: {block['rows']}. Mean training fresh-probe gain "
                  f"**{block['mean_training_gain']:+.4f}**; mean audit additional recovery over "
                  f"`H` **{block['mean_audit_additional_recovery']:+.4f}**. "
                  f"Association across arms: Pearson "
                  f"{block['correlation'].get('pearson')}, Spearman "
                  f"{block['correlation'].get('spearman')} "
                  f"(n = {block['correlation'].get('n')}).", '',
                  '_An observational association across fitted arms. It does not identify a',
                  'mechanism and no causal reading is offered._', '']
    if refresh and mech.get('refresh'):
        block = mech['refresh']
        lines += ['## Attacker refresh', '',
                  f"{block['decisions']} refresh decisions; the refreshed attacker was kept in "
                  f"**{block['refreshed_kept']}** of them. Mean monitor cross-entropy recovery "
                  f"{block['mean_recovery']:+.5f}, maximum {block['max_recovery']:+.5f}.", '',
                  'A positive recovery means the incumbent had been **fooled by the moving',
                  'channel** rather than genuinely defeated — which is exactly the failure mode',
                  'the refresh exists to detect.', '']
    if catchup:
        by_condition = defaultdict(list)
        for row in catchup:
            by_condition[row['condition']].append(float(row['catchup_gain']))
        body = [[f'`{c}`', fnum(np.mean(v)), fnum(np.max(v)), len(v)]
                for c, v in sorted(by_condition.items(),
                                   key=lambda kv: -float(np.mean(kv[1])))[:25]]
        lines += ['## Extended catch-up: budget 120 against budget 360', '',
                  'How much more the longer attack recovers on the same scope. A large value',
                  'means the shorter budget understated what was reachable.', '',
                  table(['condition', 'mean catch-up gain', 'max', 'rows'], body), '',
                  '_Full table in `MECH_CATCHUP.csv`._', '']
    lines += ['## Backing data', '', '* `MECH_TRAINING_VS_AUDIT.csv`', '* `MECH_REFRESH.csv`',
              '* `MECH_CATCHUP.csv`', '* `MECHANISM.json`', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ optimisation stability
def optimization_stability(out: Path) -> str:
    intervals = [r for r in rows_of(out / 'PAIRED_INTERVALS.csv')
                 if r['comparison_family'] == 'optimizer_stability']
    signs = [r for r in rows_of(out / 'PER_SEED_SIGNS.csv')
             if r['family'] == 'optimizer_stability']
    lines = ['# OPTIMIZATION_STABILITY — the repeat block', '',
             'The 24 repeat fits are **optimisation repetitions conditional on the three',
             'historical anchors**. They are not six or nine independent population seeds and',
             'they test only whether the proposed mechanism is stable under a different',
             'optimiser draw at the same anchor.', '']
    if not intervals:
        lines += ['_The optimiser-repeat contrasts have not been computed yet._', '']
        return '\n'.join(lines) + '\n'
    body = []
    for row in intervals:
        body.append([f"`{row['left']}` vs `{row['right']}`", row['weight'],
                     FAMILY_ENDPOINT_LABEL.get(row['endpoint'], row['endpoint']),
                     fnum(row['estimate']),
                     f"[{fnum(row['adjusted_low'])}, {fnum(row['adjusted_high'])}]",
                     'yes' if row['significantly_better'] == 'True'
                     or row['significantly_worse'] == 'True' else 'no'])
    lines += ['## Each repeat against its anchor', '',
              table(['contrast', 'weight', 'endpoint', 'estimate', 'within-contrast interval',
                     'interval excludes zero'], body), '']
    if signs:
        flips = defaultdict(set)
        for row in signs:
            flips[row['left'], row['right'], row['weight'], row['endpoint']].add(row['sign'])
        disagreeing = sum(1 for v in flips.values() if len(v) > 1)
        lines += ['## Per-seed sign agreement', '',
                  f'Of {len(flips)} repeat contrasts x endpoints x weightings, '
                  f'**{disagreeing}** have seeds disagreeing in sign.', '']
    lines += ['## Backing data', '', '* `PAIRED_INTERVALS.csv` (family `optimizer_stability`)',
              '* `PER_SEED_SIGNS.csv`', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ frontier
def frontier_analysis(out: Path) -> str:
    mech = read_json(out / 'MECHANISM.json') if (out / 'MECHANISM.json').exists() else {}
    equivalence = rows_of(out / 'MECH_BETA_EQUIVALENCE.csv')
    compression = rows_of(out / 'MECH_COMPRESSION.csv')
    trajectory = rows_of(out / 'MECH_TRAJECTORY.csv')
    per_seed = rows_of(out / 'PER_SEED.csv')

    lines = ['# FRONTIER_ANALYSIS — discrete configurations, not an interpolated curve', '',
             'Every point below is an **actual fitted release**. No realisable release is',
             'invented between two curve points.', '',
             '**No withholding mixture is used anywhere in this study.** The historical arms',
             'have a withholding grid in their own protocol; this study introduces none, makes',
             'no interpolated-release claim, and therefore claims no privacy property that would',
             'depend on hidden randomisation. Every released interface is a deterministic',
             'function of permitted inference inputs, and the release/no-release indicator does',
             'not exist here because every person in a scored pool receives the release.', '']

    # frontier points
    grouped = defaultdict(list)
    for row in per_seed:
        if (row['weight'] != 'unweighted' or row['scope'] != MAIN_SCOPE
                or row['budget'] != '360'):
            continue
        if row['kind'] == 'additional_recovery' and row['endpoint'] in FAMILY_SENSITIVE:
            grouped[row['condition'], row['endpoint']].append(float(row['value']))
        if row['kind'] == 'utility_gain_vs_H' and row['endpoint'] == 'same_residence':
            grouped[row['condition'], 'residence'].append(float(row['value']))
    conditions = sorted({c for c, _ in grouped})
    if conditions:
        body = []
        for condition in conditions:
            residence = grouped.get((condition, 'residence'))
            body.append([f'`{condition}`',
                         fnum(np.mean(residence)) if residence else '--',
                         *[fnum(np.mean(grouped[condition, e]))
                           if (condition, e) in grouped else '--' for e in FAMILY_SENSITIVE]])
        lines += ['## Residence gain over `H` against each sensitive endpoint, separately', '',
                  'Sensitive endpoints are kept **separate**. A local race cost is never',
                  'concealed behind an average coalition SEX gain.', '',
                  table(['condition', 'residence gain', *(e.split('/', 1)[1]
                                                          for e in FAMILY_SENSITIVE)], body), '']

    if equivalence:
        moved = [r for r in equivalence if r['identical_to_no_protection'] != 'True']
        identical = [r for r in equivalence if r['identical_to_no_protection'] == 'True']
        lines += ['## Is a beta arm effectively identical to beta = 0?', '',
                  'Compared on the released arrays themselves — maximum absolute difference,',
                  'norm ratio and the largest principal angle against the matched-width',
                  'no-protection channel — not on a downstream score.', '',
                  f'{len(identical)} of {len(equivalence)} arms are **bitwise identical** to',
                  'their no-protection continuation.', '']
        body = []
        for width in WIDTHS:
            for policy in POLICIES:
                for beta in BETAS:
                    rows = [r for r in equivalence
                            if r['arm'] == main_arm(width, policy, beta)]
                    if not rows:
                        continue
                    body.append([f'`{main_arm(width, policy, beta)}`',
                                 f"{np.mean([float(r['max_abs_difference']) for r in rows]):.4f}",
                                 f"{np.mean([float(r['norm_ratio']) for r in rows]):.4f}",
                                 f"{np.mean([float(r['max_principal_angle_degrees']) for r in rows]):.1f}"])
        lines += [table(['arm', 'mean max abs diff', 'mean norm ratio',
                         'mean max principal angle (deg)'], body), '']

    if compression:
        body = []
        by_condition = defaultdict(list)
        for row in compression:
            by_condition[row['condition']].append(row)
        for condition, rows in sorted(by_condition.items()):
            body.append([f'`{condition}`', rows[0]['ambient_width'],
                         f"{np.mean([float(r['numerical_rank_1e-10']) for r in rows]):.1f}",
                         f"{np.mean([float(r['effective_rank_entropy']) for r in rows]):.2f}",
                         f"{np.mean([float(r['components_for_99pct']) for r in rows]):.1f}"])
        lines += ['## Is width reduction accounting for the apparent movement?', '',
                  '**Ambient width is not effective width.** A matched ambient width with a',
                  'lower effective rank is not a width match, and is reported as such.', '',
                  table(['condition', 'ambient width', 'numerical rank', 'effective rank (exp '
                         'entropy)', 'components for 99% variance'], body), '']

    if trajectory and mech.get('distortion_versus_residence'):
        block = mech['distortion_versus_residence']
        lines += ['## Does teacher fidelity predict residence transfer at all?', '',
                  'The utility objective uses teacher distortion as a **proxy** for useful',
                  'capability. Whether that proxy tracks the reserved residence task is an open',
                  'empirical question, and it is answered descriptively here rather than',
                  'assumed.', '',
                  f"Teacher distortion against residence gain: Pearson "
                  f"{block['distortion_vs_residence'].get('pearson')}, Spearman "
                  f"{block['distortion_vs_residence'].get('spearman')} "
                  f"(n = {block['distortion_vs_residence'].get('n')}).",
                  f"Source cross-entropy against residence gain: Pearson "
                  f"{block['source_vs_residence'].get('pearson')}, Spearman "
                  f"{block['source_vs_residence'].get('spearman')}.", '',
                  '_Observational associations across fitted arms. No causal reading._', '']

    if mech.get('class_support'):
        block = mech['class_support']
        lines += ['## Unsupported protected categories, per actual fit fold', '',
                  ('Folds with at least one class absent: '
                   + (', '.join(f'`{f}`' for f in block['folds_with_an_unsupported_class'])
                      if block['folds_with_an_unsupported_class'] else '**none**')), '',
                  '**Absence in a fit split is not population absence** and is never reported as',
                  'one. Full counts in `MECH_CLASS_SUPPORT.csv`.', '']

    lines += ['## Backing data', '', '* `MECH_BETA_EQUIVALENCE.csv`', '* `MECH_COMPRESSION.csv`',
              '* `MECH_TRAJECTORY.csv`', '* `MECH_CLASS_SUPPORT.csv`', '* `PER_SEED.csv`', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ validation doc
def validation(out: Path) -> str:
    path = out / 'VALIDATION.json'
    lines = ['# VALIDATION — ten checks aimed at material risks', '',
             'Not an exhaustive test suite. Each check targets a specific way this study could',
             'silently be wrong.', '']
    if not path.exists():
        return '\n'.join(lines + ['_Validation has not been run yet._', '']) + '\n'
    report = read_json(path)
    labels = {
        'gradient_signs': 'gradient signs: stronger recovery raises the penalty, the mapper '
                          'step reduces it',
        'nested_baseline': 'nested baseline: the zero correction reproduces `p0_j` exactly',
        'label_exclusion': 'label exclusion: residence and commute unreachable from the '
                           'representation label path',
        'household_boundaries': 'household boundaries: internal folds partition '
                                '`representation_fit`, no household straddles',
        'release_parity': 'release parity: `H_A` and `H_B` bitwise preserved in every wire',
        'map_serialisation': 'map serialisation: a reloaded channel reproduces its release '
                             'bitwise',
        'resumed_unit_identity': 'resumed-unit identity: recorded hashes still match their files',
        'role_masks': 'role masks: the audited forbidden registry is the full historical eleven',
        'identical_channel_identical_endpoints': 'end-to-end: a bitwise-identical released '
                                                 'channel reproduces its comparator\'s endpoints '
                                                 'exactly under the matched-exposure scope',
        'score_aggregation': 'score aggregation: stored endpoints recomputable from stored '
                             'predictions',
        'simultaneous_construction': 'simultaneous construction: the candidate-wide family '
                                     'covers every searched contrast and dominates',
    }
    body = []
    for key, label in labels.items():
        entry = report.get(key, {})
        verdict = entry.get('pass')
        body.append([label, {True: 'PASS', False: '**FAIL**', None: 'not applicable'}[verdict]])
    lines += [table(['check', 'result'], body), '']
    signs = report.get('gradient_signs', {})
    if signs:
        lines += ['## The sign fixture, in numbers', '',
                  f"Zero-correction gain **{signs['zero_correction_gain']:+.6f}** (exactly the "
                  'service-only predictor, by construction); trained attacker gain '
                  f"**{signs['trained_attacker_gain']:+.4f}**; after 50 mapper steps under "
                  f"`U + 1.0 * penalty` the penalty falls to "
                  f"**{signs['penalty_after_mapper_steps']:+.4f}**. The two players are not "
                  'reversed and the service baseline is not being optimised in place of the '
                  'channel.', '']
    for key in ('release_parity', 'map_serialisation', 'resumed_unit_identity',
                'score_aggregation'):
        entry = report.get(key, {})
        if entry.get('pass') is False:
            lines += [f'## FAILURE in `{key}`', '', '```', str(entry), '```', '']
    lines += ['## Backing data', '', '* `VALIDATION.json`', '']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ manifest
def manifest(out: Path) -> dict:
    out = Path(out)
    # Symlinked condition directories point into the invariant study's worktree. They are
    # read-only historical evidence of ANOTHER study and are recorded by reference, never
    # hashed into this study's manifest as though they were its own output.
    linked = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_symlink())
    entries, bulk = {}, defaultdict(lambda: {'files': 0, 'bytes': 0})
    for path in sorted(out.rglob('*')):
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                    if out in parent.parents or parent == out):
            continue
        if not path.is_file():
            continue
        relative = str(path.relative_to(out))
        if relative in ('ARTIFACT_MANIFEST.json', 'HANDOFF.json'):
            continue        # non-circular: the manifest never hashes itself or the handoff
        if relative.startswith('logs/'):
            continue
        # Per-candidate fitted audit objects are bulk artifacts: thousands of small files
        # already hashed by their own `unit_complete.json`. They are summarised by count
        # and size rather than enumerated, and the unit records remain the authority.
        parts = relative.split('/')
        if 'fitted' in parts:
            key = '/'.join(parts[:parts.index('fitted') + 1])
            bulk[key]['files'] += 1
            bulk[key]['bytes'] += path.stat().st_size
            continue
        entries[relative] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    record = {
        'study': 'pcrl_direct_adversarial_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'files': entries, 'count': len(entries),
        'bulk_fitted_artifacts': {k: dict(v) for k, v in sorted(bulk.items())},
        'bulk_note': ('Per-candidate fitted audit objects are summarised by count and size. '
                      "Each role's own `unit_complete.json` already carries the per-file "
                      'hashes and is the authority for them.'),
        'symlinked_read_only_evidence': linked,
        'symlink_note': ('These names are symlinks into the invariant-baselines worktree. They '
                         "are another study's published evidence, read only, and are recorded "
                         'by reference rather than hashed into this manifest.'),
        'noncircularity': ('ARTIFACT_MANIFEST.json and HANDOFF.json are excluded from the '
                           'manifest, so no file hashes itself or a file that hashes it.'),
        'local_only': ('*.npz releases, *.pt checkpoints and *.joblib fitted objects are '
                       'gitignored by the repository policy and stay local; their hashes are '
                       'recorded in the per-unit fit records and in this manifest where they '
                       'live under the results tree.'),
    }
    write_json(out / 'ARTIFACT_MANIFEST.json', record)
    return record


def run(out: Path = OUT) -> dict:
    out = Path(out)
    written = {}
    for name, builder in (('BASELINE_TRANSPORT_COMPLETION.md', baseline_transport),
                          ('DEVELOPMENT_2018.md', development_2018),
                          ('EXPLORATORY_2017.md', exploratory_2017),
                          ('ATTACK_STRENGTH.md', attack_strength),
                          ('OPTIMIZATION_STABILITY.md', optimization_stability),
                          ('FRONTIER_ANALYSIS.md', frontier_analysis),
                          ('VALIDATION.md', validation)):
        (out / name).write_text(builder(out))
        written[name] = sha_file(out / name)
        print('WROTE', name, flush=True)
    record = manifest(out)
    print('WROTE ARTIFACT_MANIFEST.json', record['count'], 'files', flush=True)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    run(a.out)


if __name__ == '__main__':
    main()
